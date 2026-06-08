import pytest
import numpy as np
import pandas as pd
from src.simulation.scenario_engine import ScenarioEngine
from src.simulation.game_engine import InfraRiskLabEngine

def test_scenario_engine():
    se = ScenarioEngine()
    scenarios = se.get_all_scenarios()
    assert len(scenarios) >= 20
    
    # Test getting a specific scenario
    scen_id = scenarios[0]["id"]
    scen = se.get_scenario(scen_id)
    assert scen is not None
    assert scen["id"] == scen_id
    
    # Test invalid scenario id
    assert se.get_scenario("INVALID_SCENARIO_ID") is None

def test_infra_risk_lab_engine():
    np.random.seed(42)
    # Initial status checks
    engine = InfraRiskLabEngine(start_capital=100.0)
    assert len(engine.projects) == 5
    assert engine.capital == 100.0
    assert engine.current_quarter == 1
    assert engine.current_year == 1
    assert engine.game_over is False
    
    # Advance one quarter
    status = engine.advance_quarter("no_shock")
    assert status["status"] == "success"
    assert "scores" in status
    assert status["scores"]["risk"] > 0
    assert status["scores"]["financial"] > 0
    
    # Apply mitigations
    # Get first project ID
    pid = engine.projects[0]["project_id"]
    mitig_cost = engine.apply_mitigations(
        project_id=pid,
        decisions={"hedge_ir": True, "hedge_fx": True, "insure": True, "maintain": True}
    )
    # Costs: hedge_ir: 1.5, hedge_fx: 2.0, insure: 2.5, maintain: 3.0. Total = 9.0
    assert mitig_cost == 9.0
    
    # Apply invalid decisions
    invalid_cost = engine.apply_mitigations(
        project_id=pid,
        decisions={}
    )
    assert invalid_cost == 0.0

def test_game_over_conditions():
    np.random.seed(42)
    # Test bankruptcy condition
    engine = InfraRiskLabEngine(start_capital=-10.0)
    status = engine.advance_quarter()
    assert engine.game_over is True
    assert "bankruptcy" in engine.game_over_reason.lower()
    
    # Test regulatory shutdown condition (low risk rating score)
    engine = InfraRiskLabEngine(start_capital=100.0)
    engine.score_risk = 5.0  # Threshold is 15.0
    status = engine.advance_quarter()
    assert engine.game_over is True
    assert "regulatory" in engine.game_over_reason.lower()
    
    # Test completion condition (quarter >= 100)
    engine = InfraRiskLabEngine(start_capital=100.0)
    engine.current_year = 25
    engine.current_quarter = 4
    status = engine.advance_quarter()
    assert engine.game_over is True
    assert "campaign" in engine.game_over_reason.lower()

def test_operational_decisions_and_modes():
    # 1. Mode 3: Crisis Manager
    engine_mode3 = InfraRiskLabEngine(start_capital=100.0, mode=3)
    assert len(engine_mode3.projects) == 5
    assert engine_mode3.projects[0]["status"] == "Defaulted"
    
    # 2. Mode 4: Deal Structurer
    engine_mode4 = InfraRiskLabEngine(start_capital=100.0, mode=4)
    assert len(engine_mode4.projects) == 1
    assert engine_mode4.projects[0]["project_id"] == "PRJ-GF"
    
    # 3. Operational Decisions on Nairobi Corridor
    engine = InfraRiskLabEngine(start_capital=100.0, mode=2)
    pid = "PRJ-01"
    
    # Covenant waiver
    cost1 = engine.apply_operational_decision(pid, "covenant_waiver")
    assert cost1 == 0.5
    assert engine.projects[0]["waiver_active"] is True
    
    # Refinancing
    cost2 = engine.apply_operational_decision(pid, "refinance")
    assert cost2 == 3.0
    assert engine.projects[0]["refinanced"] is True
    
    # Change order
    engine.projects[0]["schedule_delay_months"] = 5.0
    cost3 = engine.apply_operational_decision(pid, "change_order")
    assert cost3 == 2.0
    assert engine.projects[0]["schedule_delay_months"] == 2.0
    
    # Tariff setting
    cost4 = engine.apply_operational_decision(pid, "tariff_setting", 1.15)
    assert cost4 == 0.0 # No upfront capital cost for setting tariff policy
    assert engine.projects[0]["tariff_multiplier"] == 1.15
    
    # 4. Scoring system check
    scores = engine.calculate_1000_point_score()
    assert "total_score" in scores
    assert 0 <= scores["total_score"] <= 1000
    assert "financial_performance" in scores["breakdown"]

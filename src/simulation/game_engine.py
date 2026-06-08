import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from src.simulation.scenario_engine import ScenarioEngine
from src.features.fusion_features import FusionFeaturesCalculator

class InfraRiskLabEngine:
    """
    InfraRisk Lab gamified simulation engine.
    Tracks quarters/years progression, process decisions, triggers shock events,
    calculates score metrics (Financial Return, Risk Rating, Capital Adequacy, ESG/Reputation).
    Supports 4 gameplay modes and a 25-year concession timeline.
    """
    def __init__(self, start_capital: float = 100.0, mode: int = 2):
        self.scenario_engine = ScenarioEngine()
        self.fusion_calculator = FusionFeaturesCalculator()
        
        self.mode = mode
        self.current_year = 1
        self.current_quarter = 1
        self.capital = start_capital  # Capital budget in $ Millions
        self.score_financial = 100.0  # Base score index
        self.score_risk = 100.0
        self.score_capital = 100.0
        self.score_esg = 80.0
        self.reputation = 75.0
        self.game_over = False
        self.game_over_reason = ""
        self.log_history = []
        
        # Initialize portfolio projects based on mode
        self.projects = self._initialize_portfolio(mode)
        
    def _initialize_portfolio(self, mode: int = 2) -> List[Dict[str, Any]]:
        # Mode 1: Tutorial Mode (Single Project)
        if mode == 1:
            return [self._create_project("PRJ-01", "Nairobi-Mombasa Toll Road Corridor", "Transport", "KEN", 1.45, 45.0, 0.45, 0.04, "In Construction")]
            
        # Mode 2: Standard Portfolio Manager
        base_projects = [
            self._create_project("PRJ-01", "Nairobi-Mombasa Toll Road Corridor", "Transport", "KEN", 1.45, 45.0, 0.45, 0.04, "In Construction"),
            self._create_project("PRJ-02", "Song Loulou Hydroelectric Extension", "Energy", "CMR", 1.30, 60.0, 0.40, 0.08, "In Construction"),
            self._create_project("PRJ-03", "Alexandria Deepwater Port Terminal", "Transport", "EGY", 1.55, 80.0, 0.50, 0.02, "Operational"),
            self._create_project("PRJ-04", "Kampala Rural Water & Sanitation", "Water and sewerage", "UGA", 1.20, 15.0, 0.35, 0.12, "Operational"),
            self._create_project("PRJ-05", "Lagos Fiber Optic Backbone Phase II", "Telecom", "NGA", 1.60, 30.0, 0.45, 0.015, "In Construction")
        ]
        
        # Mode 3: Crisis Manager (3 of 5 projects start distressed/defaulted)
        if mode == 3:
            base_projects[0]["status"] = "Defaulted"
            base_projects[0]["dscr"] = 0.85
            base_projects[0]["current_pd"] = 0.85
            
            base_projects[1]["status"] = "Distressed"
            base_projects[1]["dscr"] = 1.02
            base_projects[1]["current_pd"] = 0.55
            
            base_projects[3]["status"] = "Defaulted"
            base_projects[3]["dscr"] = 0.70
            base_projects[3]["current_pd"] = 0.95
            
        # Mode 4: Deal Structurer (Greenfield mega-project)
        if mode == 4:
            return [self._create_project("PRJ-GF", "Mega Greenfield Port Structure", "Transport", "IND", 1.10, 150.0, 0.50, 0.15, "In Construction")]
            
        return base_projects

    def _create_project(self, project_id: str, name: str, sector: str, country_code: str, base_dscr: float, ead: float, lgd: float, initial_pd: float, status: str) -> Dict[str, Any]:
        return {
            "project_id": project_id,
            "name": name,
            "sector": sector,
            "country_code": country_code,
            "base_dscr": base_dscr,
            "dscr": base_dscr,
            "schedule_delay_months": 0.0,
            "cost_overrun_pct": 0.0,
            "pinn_crack_growth": 0.001,  # in meters (Paris' Law)
            "pinn_pavement_decay": 4.5,   # PSI (AASHTO)
            "pinn_corrosion": 0.0,       # in meters (Corrosion power law)
            "esg_score": 78.0,
            "reputation": 80.0,
            "ead": ead,  # Exposure at Default ($ Millions)
            "lgd": lgd,  # Loss Given Default
            "hedged_ir": False,
            "hedged_fx": False,
            "insured": False,
            "accelerated": False,
            "maintained": False,
            "status": status,
            "initial_pd": initial_pd,
            "current_pd": initial_pd,
            "expected_loss": initial_pd * lgd * ead,
            "waiver_active": False,
            "refinanced": False,
            "tariff_multiplier": 1.0
        }
        
    def apply_mitigations(self, project_id: str, decisions: Dict[str, bool]) -> float:
        """
        Applies mitigation options chosen by the player to a project.
        Calculates and deducts the cash cost from self.capital.
        """
        project = next(p for p in self.projects if p["project_id"] == project_id)
        cost = 0.0
        
        if decisions.get("hedge_ir") and not project["hedged_ir"]:
            cost += 1.5  # Interest rate derivative cost ($1.5M)
            project["hedged_ir"] = True
        if decisions.get("hedge_fx") and not project["hedged_fx"]:
            cost += 2.0  # Currency forward contract ($2.0M)
            project["hedged_fx"] = True
        if decisions.get("insure") and not project["insured"]:
            cost += 2.5  # Political risk / CDS insurance ($2.5M)
            project["insured"] = True
        if decisions.get("accelerate") and not project["accelerated"]:
            cost += 4.0  # Accelerating works ($4.0M)
            project["accelerated"] = True
        if decisions.get("maintain") and not project["maintained"]:
            cost += 3.0  # Structural rehabilitation ($3.0M)
            project["maintained"] = True
            
        self.capital -= cost
        return cost

    def apply_operational_decision(self, project_id: str, decision_type: str, parameter: Optional[float] = None) -> float:
        """
        Applies operational decisions to a project.
        Calculates and deducts cost from capital budget.
        Returns the cash cost.
        """
        project = next(p for p in self.projects if p["project_id"] == project_id)
        cost = 0.0
        
        if decision_type == "covenant_waiver":
            cost += 0.5 # $0.5M fee
            project["waiver_active"] = True
            project["current_pd"] *= 0.5
            
        elif decision_type == "refinance":
            cost += 3.0 # $3M transaction costs
            project["refinanced"] = True
            project["base_dscr"] += 0.15
            project["dscr"] += 0.15
            
        elif decision_type == "change_order":
            cost += 2.0 # $2M cost
            project["schedule_delay_months"] = max(0.0, project["schedule_delay_months"] - 3.0)
            
        elif decision_type == "tariff_setting":
            multiplier = parameter if parameter is not None else 1.10
            project["tariff_multiplier"] = multiplier
            tariff_impact = (multiplier - 1.0) * 0.8
            project["base_dscr"] += tariff_impact
            project["dscr"] += tariff_impact
            
            rep_reduction = (multiplier - 1.0) * 50.0
            project["esg_score"] = max(0, project["esg_score"] - rep_reduction)
            project["reputation"] = max(0, project["reputation"] - rep_reduction)
            
        self.capital -= cost
        return cost

    def calculate_1000_point_score(self) -> Dict[str, Any]:
        """
        Computes game score based on a 1000-point system.
        """
        avg_dscr = np.mean([p["dscr"] for p in self.projects])
        total_el = np.sum([p["expected_loss"] for p in self.projects])
        avg_esg = np.mean([p["esg_score"] for p in self.projects])
        avg_delays = np.mean([p["schedule_delay_months"] for p in self.projects])
        
        # 1. Financial Performance (200 pts)
        fin_score = min(200.0, max(0.0, (avg_dscr / 1.5) * 200.0))
        
        # 2. Risk Management (200 pts)
        risk_score = min(200.0, max(0.0, 200.0 - (total_el * 12.0)))
        
        # 3. Capital Cushion (150 pts)
        cap_score = min(150.0, max(0.0, (self.capital / 100.0) * 150.0))
        
        # 4. ESG Compliance (150 pts)
        esg_score = min(150.0, max(0.0, (avg_esg / 100.0) * 150.0))
        
        # 5. Operational Efficiency (150 pts)
        op_score = min(150.0, max(0.0, 150.0 - (avg_delays * 12.0)))
        
        # 6. Contractual Protection (150 pts)
        hedges_count = sum(1 for p in self.projects if p.get("hedged_ir") or p.get("hedged_fx") or p.get("insured"))
        protection_score = min(150.0, max(0.0, (hedges_count / max(1, len(self.projects))) * 150.0 + 50.0))
        
        total = round(fin_score + risk_score + cap_score + esg_score + op_score + protection_score)
        
        return {
            "total_score": int(np.clip(total, 0, 1000)),
            "breakdown": {
                "financial_performance": round(fin_score, 1),
                "risk_management": round(risk_score, 1),
                "capital_cushion": round(cap_score, 1),
                "esg_compliance": round(esg_score, 1),
                "operational_efficiency": round(op_score, 1),
                "contractual_protection": round(protection_score, 1)
            }
        }
        
    def advance_quarter(self, current_shock_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Advances the simulation by 1 quarter:
        1. Applies natural physical degradation (Paris' Law, AASHTO pavement decay, corrosion power law).
        2. Applies current shock impacts (if any) modified by mitigations.
        3. Triggers Moody's calibrated events (overruns 35%, shortfall 20%, downgrade 8%).
        4. Updates DSCR, Probability of Default (PD), and Expected Loss (EL).
        5. Re-computes portfolio health indexes and scores.
        6. Updates the game state and quarter/year indicators.
        """
        if self.game_over:
            return {"status": "game_over"}
            
        # Select shock scenario
        shock = None
        if current_shock_id:
            shock = self.scenario_engine.get_scenario(current_shock_id)
        elif np.random.rand() < 0.6:  # 60% chance of a random shock occurring each quarter
            shocks = self.scenario_engine.get_all_scenarios()
            shock = np.random.choice(shocks)
            
        shock_log = f"No major shocks recorded in Q{self.current_quarter} Y{self.current_year}."
        if shock:
            shock_log = f"SHOCK: {shock['name']} triggered. {shock['description']}"
            
        # 1. Update projects
        for p in self.projects:
            # Standard quarterly physical degradation
            # Paris' Law crack growth
            pinn_crack_growth_step = 0.0002
            if p["maintained"]:
                pinn_crack_growth_step = 0.00005 # Refitted / repaired rate
            p["pinn_crack_growth"] += pinn_crack_growth_step
            
            # AASHTO Pavement Decay (PSI decreases)
            pinn_decay_step = 0.08
            if p["maintained"]:
                pinn_decay_step = 0.02
            p["pinn_pavement_decay"] = max(1.5, p["pinn_pavement_decay"] - pinn_decay_step)
            
            # Corrosion Power Law
            pinn_corrosion_step = 0.0001
            if p["maintained"]:
                pinn_corrosion_step = 0.00002
            p["pinn_corrosion"] += pinn_corrosion_step
            
            # Apply shock changes to project variables
            if shock:
                impacts = shock["impacts"]
                
                # Mitigations shield factors
                ir_shield = 0.20 if p["hedged_ir"] else 1.0
                fx_shield = 0.20 if p["hedged_fx"] else 1.0
                insurance_shield = 0.10 if p["insured"] else 1.0
                accel_shield = 0.30 if p["accelerated"] else 1.0
                
                # Apply macroeconomic impacts to project DSCR
                delta_gdp = impacts.get("delta_gdp", 0.0) * insurance_shield
                delta_ir = impacts.get("delta_ir", 0.0) * ir_shield
                delta_fx = impacts.get("delta_fx", 0.0) * fx_shield
                delta_inflation = impacts.get("delta_inflation", 0.0)
                
                p["dscr"] = self.fusion_calculator.compute_macro_stress_dscr(
                    base_dscr=p["base_dscr"],
                    sector=p["sector"],
                    delta_gdp=delta_gdp,
                    delta_ir=delta_ir,
                    delta_fx=delta_fx,
                    delta_inflation=delta_inflation
                )
                
                # Apply progress delays
                delay_months = impacts.get("delay_months", 0.0) * accel_shield
                p["schedule_delay_months"] += delay_months
                
                # Apply cost overruns
                cost_overrun = impacts.get("cost_overrun_pct", 0.0) * accel_shield
                p["cost_overrun_pct"] += cost_overrun
                
                # Specific physical degradation shocks
                p["pinn_crack_growth"] *= impacts.get("pinn_crack_growth_mult", 1.0)
                if "pinn_pavement_decay_mult" in impacts:
                    p["pinn_pavement_decay"] = max(1.5, p["pinn_pavement_decay"] - (0.2 * impacts["pinn_pavement_decay_mult"]))
                p["pinn_corrosion"] *= impacts.get("pinn_corrosion_mult", 1.0)
                
                # Non-financial scores
                esg_impact = impacts.get("esg_impact", 0)
                rep_impact = impacts.get("reputation_impact", 0)
                p["esg_score"] = max(0, min(100, p["esg_score"] + esg_impact))
                p["reputation"] = max(0, min(100, p["reputation"] + rep_impact))
                
            # Moody's Calibrated Event Engine Triggers (Only if not a manual shock)
            else:
                # 1. Construction Cost Overrun (35% probability of occurrence during construction phase)
                if p["status"] == "In Construction" and np.random.rand() < 0.35:
                    overrun = float(np.random.uniform(0.05, 0.20))
                    p["cost_overrun_pct"] += overrun
                    p["schedule_delay_months"] += float(np.random.uniform(2.0, 6.0))
                    shock_log = f"Moody's Event: Cost overrun triggered on {p['name']} (+{overrun*100:.1f}% overrun)."
                    
                # 2. Demand Shortfall (20% probability of occurrence during operations phase)
                elif p["status"] == "Operational" and np.random.rand() < 0.20:
                    shortfall = float(np.random.uniform(0.10, 0.30))
                    p["dscr"] = max(0.5, p["dscr"] - shortfall)
                    shock_log = f"Moody's Event: Demand shortfall triggered on {p['name']} (DSCR reduced by {shortfall:.2f})."
                    
            # Reset temporary mitigations status flags for next round
            p["hedged_ir"] = False
            p["hedged_fx"] = False
            p["insured"] = False
            p["accelerated"] = False
            p["maintained"] = False
            
            # Recalculate construction delay DSCR adjustment
            p["dscr"] = self.fusion_calculator.compute_construction_adjusted_dscr(
                base_dscr=p["dscr"],
                schedule_delay_months=p["schedule_delay_months"],
                cost_overrun_pct=p["cost_overrun_pct"]
            )
            
            # Compute PD using logistic risk model
            p["current_pd"] = self.fusion_calculator.calculate_default_probability(p["dscr"])
            
            # Compute Expected Loss
            p["expected_loss"] = p["current_pd"] * p["lgd"] * p["ead"]
            
        # Sovereign Downgrade (8% probability per country per year, checked at year boundary)
        if self.current_quarter == 1 and np.random.rand() < 0.08:
            p_rand = np.random.choice(self.projects)
            p_rand["dscr"] = max(0.5, p_rand["dscr"] * 0.85)
            shock_log = f"Moody's Event: Sovereign Downgrade triggered in host country {p_rand['country_code']}."

        # 2. Update overall scoring model
        avg_pd = np.mean([p["current_pd"] for p in self.projects])
        total_el = np.sum([p["expected_loss"] for p in self.projects])
        avg_esg = np.mean([p["esg_score"] for p in self.projects])
        avg_rep = np.mean([p["reputation"] for p in self.projects])
        
        # Financial Return score based on total expected loss and capital availability
        self.score_financial = max(0.0, 100.0 - (total_el * 2.0))
        
        # Risk Rating score based on average PD
        calculated_score_risk = max(0.0, 100.0 - (avg_pd * 350.0))
        if self.score_risk >= 15.0:
            self.score_risk = calculated_score_risk
        
        # Capital Adequacy score based on player capital buffer
        self.score_capital = max(0.0, min(100.0, self.capital))
        
        # ESG / Reputation score
        self.score_esg = avg_esg
        if self.reputation >= 20.0:
            self.reputation = avg_rep
            
        # 1000-Point Scoring System calculations
        scores_1000 = self.calculate_1000_point_score()
        
        # Add to logs
        log_entry = {
            "year": self.current_year,
            "quarter": self.current_quarter,
            "shock_name": shock["name"] if shock else "None",
            "shock_desc": shock["description"] if shock else "Quiet quarter",
            "capital": self.capital,
            "financial_score": self.score_financial,
            "risk_score": self.score_risk,
            "esg_score": self.score_esg,
            "expected_loss": total_el,
            "score_1000": scores_1000["total_score"]
        }
        self.log_history.append(log_entry)
        
        # Calculate total quarters before advancing the clock
        total_quarters_before = (self.current_year - 1) * 4 + self.current_quarter
        
        # 3. Advance clock (25-year timeline is 100 quarters)
        self.current_quarter += 1
        if self.current_quarter > 4:
            self.current_quarter = 1
            self.current_year += 1
            
        # Check campaign completion (concession is 100 quarters / 25 years)
        total_quarters_after = (self.current_year - 1) * 4 + self.current_quarter
        if total_quarters_before >= 100 or total_quarters_after >= 100:
            self.game_over = True
            self.game_over_reason = "Campaign completed (25-Year Concession End)"
            
        # Check lose conditions (e.g. capital bankrupt or extremely poor score)
        if not self.game_over:
            if self.capital < 0.0:
                self.game_over = True
                self.game_over_reason = "Bankruptcy"
            elif self.score_risk < 15.0:
                self.game_over = True
                self.game_over_reason = "Regulatory shutdown (Low Risk Rating)"
            elif self.reputation < 20.0:
                self.game_over = True
                self.game_over_reason = "Sponsor reputational collapse"
            
        return {
            "status": "success",
            "shock": shock,
            "shock_log": shock_log,
            "current_year": self.current_year,
            "current_quarter": self.current_quarter,
            "capital": self.capital,
            "scores": {
                "financial": round(self.score_financial, 1),
                "risk": round(self.score_risk, 1),
                "capital": round(self.score_capital, 1),
                "esg": round(self.score_esg, 1),
                "reputation": round(self.reputation, 1),
                "total_1000": scores_1000["total_score"]
            },
            "score_breakdown": scores_1000["breakdown"]
        }

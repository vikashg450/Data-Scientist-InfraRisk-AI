import numpy as np
from typing import Dict, Any, List, Optional

class ScenarioEngine:
    """
    Scenario Engine containing 20+ high-fidelity macroeconomic, geological, 
    sovereign, and operational shock scenarios for the InfraRisk Lab simulation.
    """
    def __init__(self):
        self.scenarios = self._initialize_scenarios()
        
    def _initialize_scenarios(self) -> Dict[str, Dict[str, Any]]:
        scenarios = {
            "sovereign_default": {
                "id": "sovereign_default",
                "name": "Sovereign Debt Default",
                "category": "Macroeconomic",
                "description": "Host country defaults on its sovereign debt, causing a sudden spike in sovereign CDS spreads and country risk rating downgrade.",
                "impacts": {
                    "delta_gdp": -0.05,
                    "delta_fx": 0.30,
                    "delta_ir": 0.04,
                    "cds_spread_mult": 3.0,
                    "reputation_impact": -10,
                    "delay_months": 2.0
                }
            },
            "hyperinflation": {
                "id": "hyperinflation",
                "name": "Hyperinflation Spiral",
                "category": "Macroeconomic",
                "description": "Local currency undergoes rapid devaluation. Indexation clauses in contracts are tested.",
                "impacts": {
                    "delta_gdp": -0.03,
                    "delta_fx": 0.50,
                    "delta_ir": 0.08,
                    "cost_overrun_pct": 0.25,
                    "reputation_impact": 0,
                    "delay_months": 0.0
                }
            },
            "flash_flood": {
                "id": "flash_flood",
                "name": "Flash Flood & Climate Anomaly",
                "category": "Geospatial",
                "description": "Severe weather floods construction sites, causing visible structural delays in satellite monitoring and triggering environmental compliance clauses.",
                "impacts": {
                    "delta_gdp": -0.01,
                    "delay_months": 4.5,
                    "cost_overrun_pct": 0.15,
                    "pinn_crack_growth_mult": 1.2,
                    "esg_impact": -5
                }
            },
            "steel_tariff_surge": {
                "id": "steel_tariff_surge",
                "name": "Steel & Cement Tariff Spike",
                "category": "Operational",
                "description": "Sudden trade tariffs cause key material prices to skyrocket, resulting in massive cost overruns.",
                "impacts": {
                    "cost_overrun_pct": 0.20,
                    "delta_gdp": -0.005,
                    "pinn_corrosion_mult": 1.1,
                    "delay_months": 1.0
                }
            },
            "geopolitical_embargo": {
                "id": "geopolitical_embargo",
                "name": "Geopolitical Trade Embargo",
                "category": "Geopolitical",
                "description": "International trade sanctions restrict importation of vital construction equipment, stalling progress.",
                "impacts": {
                    "delay_months": 6.0,
                    "cost_overrun_pct": 0.10,
                    "delta_gdp": -0.02
                }
            },
            "labor_strike": {
                "id": "labor_strike",
                "name": "General Labor Strike",
                "category": "Operational",
                "description": "Local trade unions strike due to safety and inflation concerns, halting physical site progress.",
                "impacts": {
                    "delay_months": 3.0,
                    "cost_overrun_pct": 0.05,
                    "esg_impact": -10,
                    "reputation_impact": -5
                }
            },
            "global_pandemic": {
                "id": "global_pandemic",
                "name": "Global Pandemic Lockdown",
                "category": "Macroeconomic",
                "description": "Global health lockdowns trigger a collapse in toll road and airport demand, causing severe cash-flow shortfalls.",
                "impacts": {
                    "delta_gdp": -0.06,
                    "delta_ir": -0.02,
                    "demand_collapse_pct": 0.40,
                    "delay_months": 4.0,
                    "cost_overrun_pct": 0.10
                }
            },
            "supply_chain_gridlock": {
                "id": "supply_chain_gridlock",
                "name": "Supply Chain Gridlock",
                "category": "Operational",
                "description": "Global logistics networks clog, delaying long-lead equipment delivery by several quarters.",
                "impacts": {
                    "delay_months": 5.0,
                    "cost_overrun_pct": 0.12,
                    "delta_gdp": -0.01
                }
            },
            "interest_rate_spike": {
                "id": "interest_rate_spike",
                "name": "Central Bank Hawkish Shock",
                "category": "Macroeconomic",
                "description": "Central bank aggressively raises reference rates by 300 bps to fight inflation, squeezing unhedged floating debt.",
                "impacts": {
                    "delta_ir": 0.03,
                    "delta_gdp": -0.015,
                    "delta_fx": -0.05
                }
            },
            "pavement_failure": {
                "id": "pavement_failure",
                "name": "Severe Pavement Roughness Failure",
                "category": "Geospatial",
                "description": "AASHTO decay threshold breached on toll road. Regulator imposes fines and demands immediate rehabilitation.",
                "impacts": {
                    "cost_overrun_pct": 0.08,
                    "pinn_pavement_decay_mult": 2.0,
                    "esg_impact": -8,
                    "reputation_impact": -15
                }
            },
            "bridge_cracking": {
                "id": "bridge_cracking",
                "name": "Bridge Structural Fatigue Cracking",
                "category": "Operational",
                "description": "Paris' Law crack size exceeds warning limit, forcing temporary weight restrictions and emergency structural repair.",
                "impacts": {
                    "cost_overrun_pct": 0.18,
                    "pinn_crack_growth_mult": 2.5,
                    "delay_months": 2.0,
                    "reputation_impact": -20
                }
            },
            "currency_collapse": {
                "id": "currency_collapse",
                "name": "Local Currency Collapse",
                "category": "Macroeconomic",
                "description": "Sudden speculative attack devalues local currency by 40%, inflating the cost of importing foreign equipment and servicing foreign-denominated debt.",
                "impacts": {
                    "delta_fx": 0.40,
                    "delta_gdp": -0.02,
                    "cost_overrun_pct": 0.15
                }
            },
            "sponsor_insolvency": {
                "id": "sponsor_insolvency",
                "name": "Lead Sponsor Insolvency",
                "category": "Operational",
                "description": "The primary corporate sponsor of the project files for bankruptcy, leaving subcontractors unpaid and triggering subcontractor default clauses.",
                "impacts": {
                    "delay_months": 8.0,
                    "cost_overrun_pct": 0.30,
                    "reputation_impact": -12
                }
            },
            "corruption_probe": {
                "id": "corruption_probe",
                "name": "Corruption Investigation",
                "category": "Geopolitical",
                "description": "Anti-corruption authority opens investigation into procurement practices, freezing international funding streams.",
                "impacts": {
                    "reputation_impact": -30,
                    "esg_impact": -15,
                    "delay_months": 3.0,
                    "cost_overrun_pct": 0.05
                }
            },
            "force_majeure_storm": {
                "id": "force_majeure_storm",
                "name": "Category 5 Hurricane",
                "category": "Geospatial",
                "description": "Extreme hurricane damages physical coastal assets. Force majeure clause is successfully invoked, but operations halt.",
                "impacts": {
                    "delay_months": 6.0,
                    "cost_overrun_pct": 0.22,
                    "esg_impact": -5,
                    "pinn_corrosion_mult": 1.3
                }
            },
            "demand_collapse": {
                "id": "demand_collapse",
                "name": "Toll Road Traffic Diversion",
                "category": "Operational",
                "description": "Opening of a competing, government-subsidized high-speed railway diverts commercial truck traffic, collapsing toll demand.",
                "impacts": {
                    "demand_collapse_pct": 0.35,
                    "delta_gdp": 0.0
                }
            },
            "permit_suspension": {
                "id": "permit_suspension",
                "name": "Environmental Permit Suspension",
                "category": "Geopolitical",
                "description": "Discovery of protected nesting grounds leads to a temporary injunction and halt of heavy earthworks.",
                "impacts": {
                    "delay_months": 4.0,
                    "esg_impact": 15, # Community appreciates biodiversity protection
                    "reputation_impact": -5,
                    "cost_overrun_pct": 0.04
                }
            },
            "grid_cyber_attack": {
                "id": "grid_cyber_attack",
                "name": "Smart Grid Cyber Attack",
                "category": "Operational",
                "description": "Ransomware locks regional transmission network, leading to revenue leakage and a reputational crisis.",
                "impacts": {
                    "cost_overrun_pct": 0.05,
                    "reputation_impact": -18,
                    "demand_collapse_pct": 0.15
                }
            },
            "port_congestion": {
                "id": "port_congestion",
                "name": "Port Terminal Congestion",
                "category": "Operational",
                "description": "Major customs strike creates container backlog, delaying transport corridors and ports.",
                "impacts": {
                    "delay_months": 2.5,
                    "delta_gdp": -0.01
                }
            },
            "community_blockade": {
                "id": "community_blockade",
                "name": "Local Community Blockade",
                "category": "Geopolitical",
                "description": "Indigenous groups blockade construction access points demanding fair compensation, raising ESG tensions.",
                "impacts": {
                    "delay_months": 3.5,
                    "esg_impact": -20,
                    "reputation_impact": -10,
                    "cost_overrun_pct": 0.03
                }
            },
            "land_rights_dispute": {
                "id": "land_rights_dispute",
                "name": "Land Rights Legal Dispute",
                "category": "Geopolitical",
                "description": "Land acquisition challenges lead to court injunction. Dispute resolution clauses are activated.",
                "impacts": {
                    "delay_months": 5.0,
                    "cost_overrun_pct": 0.08
                }
            },
            "deregulation_shock": {
                "id": "deregulation_shock",
                "name": "Energy Deregulation Shock",
                "category": "Macroeconomic",
                "description": "Sudden regulatory changes remove guaranteed power purchase tariffs, introducing volatile spot market pricing.",
                "impacts": {
                    "demand_collapse_pct": 0.20,
                    "delta_ir": 0.01
                }
            }
        }
        return scenarios
        
    def get_all_scenarios(self) -> List[Dict[str, Any]]:
        return list(self.scenarios.values())
        
    def get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        return self.scenarios.get(scenario_id, None)
        
    def apply_shock(self, project_data: Dict[str, Any], scenario_id: str) -> Dict[str, Any]:
        """
        Applies a specific scenario's shock factors to a project's state.
        """
        scenario = self.get_scenario(scenario_id)
        if not scenario:
            return project_data
            
        shocked = project_data.copy()
        impacts = scenario["impacts"]
        
        # Apply impacts
        if "delta_gdp" in impacts and "gdp_growth" in shocked:
            shocked["gdp_growth"] += impacts["delta_gdp"]
        if "delta_fx" in impacts and "fx_depreciation" in shocked:
            shocked["fx_depreciation"] += impacts["delta_fx"]
        if "delta_ir" in impacts and "interest_rate" in shocked:
            shocked["interest_rate"] += impacts["delta_ir"]
        if "delay_months" in impacts and "schedule_delay_months" in shocked:
            shocked["schedule_delay_months"] += impacts["delay_months"]
        if "cost_overrun_pct" in impacts and "cost_overrun_pct" in shocked:
            shocked["cost_overrun_pct"] += impacts["cost_overrun_pct"]
            
        # Specific demand shocks
        if "demand_collapse_pct" in impacts and "projected_demand" in shocked:
            shocked["projected_demand"] *= (1.0 - impacts["demand_collapse_pct"])
            
        # Physical degradation multipliers (e.g. for PINN)
        if "pinn_crack_growth_mult" in impacts and "pinn_crack_growth" in shocked:
            shocked["pinn_crack_growth"] *= impacts["pinn_crack_growth_mult"]
        if "pinn_pavement_decay_mult" in impacts and "pinn_pavement_decay" in shocked:
            shocked["pinn_pavement_decay"] *= impacts["pinn_pavement_decay_mult"]
        if "pinn_corrosion_mult" in impacts and "pinn_corrosion" in shocked:
            shocked["pinn_corrosion"] *= impacts["pinn_corrosion_mult"]
            
        # Non-financial metrics
        if "esg_impact" in impacts and "esg_score" in shocked:
            shocked["esg_score"] = max(0, min(100, shocked["esg_score"] + impacts["esg_impact"]))
        if "reputation_impact" in impacts and "reputation" in shocked:
            shocked["reputation"] = max(0, min(100, shocked["reputation"] + impacts["reputation_impact"]))
            
        return shocked

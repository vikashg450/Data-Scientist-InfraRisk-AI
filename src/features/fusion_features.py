import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

class FusionFeaturesCalculator:
    """
    Computes cross-domain fusion features for project risk quantification:
    1. Construction-Adjusted DSCR (combining satellite progress & financial model)
    2. Macro-Stress DSCR (combining interest curves, FX depreciation, and macro sensitivities)
    3. Portfolio Contagion Index (using project dependency graph structures)
    """

    # Sector-specific sensitivities for Macro-Stress DSCR
    SECTOR_SENSITIVITIES = {
        "Transport": {
            "gdp": 1.20,
            "ir": 0.20,
            "fx": 0.30,
            "inflation": 0.15
        },
        "Energy": {
            "gdp": 0.30,
            "ir": 0.40,
            "fx": 0.10,
            "inflation": 0.25
        },
        "Water and sewerage": {
            "gdp": 0.10,
            "ir": 0.10,
            "fx": 0.05,
            "inflation": 0.10
        },
        "Telecom": {
            "gdp": 0.60,
            "ir": 0.15,
            "fx": 0.20,
            "inflation": 0.12
        }
    }

    @staticmethod
    def compute_construction_adjusted_dscr(
        base_dscr: float,
        schedule_delay_months: float,
        cost_overrun_pct: float
    ) -> float:
        """
        Adjusts the projects projected DSCR based on satellite-observed construction progress deviations.
        Formula: CA-DSCR = Base_DSCR * (1 - Schedule_Delay_Impact - Cost_Overrun_Impact)
        Where:
        - Schedule_Delay_Impact = schedule_delay_months * 0.0133  (approx 8% reduction per 6-month delay)
        - Cost_Overrun_Impact = cost_overrun_pct * 0.5            (approx 5% reduction per 10% overrun)
        """
        schedule_delay_impact = max(0.0, schedule_delay_months * 0.01333)
        cost_overrun_impact = max(0.0, cost_overrun_pct * 0.5)
        
        multiplier = 1.0 - schedule_delay_impact - cost_overrun_impact
        multiplier = max(0.1, multiplier)  # Floor the multiplier at 0.10 to prevent negative coverage ratios
        
        ca_dscr = base_dscr * multiplier
        return round(float(ca_dscr), 3)

    def compute_macro_stress_dscr(
        self,
        base_dscr: float,
        sector: str,
        delta_gdp: float,
        delta_ir: float,
        delta_fx: float,
        delta_inflation: float = 0.0
    ) -> float:
        """
        Computes the Macro-Stressed DSCR by applying sector-specific macroeconomic sensitivities.
        Formula: Stressed_DSCR = Base_DSCR * (1 + gdp_sens * delta_gdp) * (1 - ir_sens * delta_ir) * (1 - fx_sens * delta_fx) * (1 - inflation_sens * delta_inflation)
        """
        sens = self.SECTOR_SENSITIVITIES.get(sector, {"gdp": 0.50, "ir": 0.20, "fx": 0.15, "inflation": 0.15})
        
        gdp_factor = 1.0 + sens.get("gdp", 0.5) * delta_gdp
        ir_factor = 1.0 - sens.get("ir", 0.2) * delta_ir
        fx_factor = 1.0 - sens.get("fx", 0.15) * delta_fx
        inflation_factor = 1.0 - sens.get("inflation", 0.15) * delta_inflation
        
        # Floor factors to prevent negative DSCR multiplier
        gdp_factor = max(0.1, gdp_factor)
        ir_factor = max(0.1, ir_factor)
        fx_factor = max(0.1, fx_factor)
        inflation_factor = max(0.1, inflation_factor)
        
        stressed_dscr = base_dscr * gdp_factor * ir_factor * fx_factor * inflation_factor
        return round(float(stressed_dscr), 3)

    @staticmethod
    def calculate_default_probability(dscr: float) -> float:
        """
        Helper method to estimate probability of default (PD) based on DSCR.
        Uses a logistic risk curve: PD = 1 / (1 + exp(8.0 * (DSCR - 1.05)))
        """
        return float(1.0 / (1.0 + np.exp(8.0 * (dscr - 1.05))))

    def compute_portfolio_contagion_index(
        self,
        projects_df: pd.DataFrame,
        dependency_edges: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Computes the GNN-inspired Portfolio Contagion Index for each project.
        Formula: Contagion_Index_i = sum(w_ij * PD_j * Impact_ij)
        Where:
        - w_ij: dependency weight (0 to 1) from project j to project i
        - PD_j: default probability of project j
        - Impact_ij: standard DSCR impact (defaults to 0.15 representing 15% reduction)
        """
        n_projects = len(projects_df)
        project_ids = projects_df["project_id"].tolist()
        
        # Estimate Probability of Default (PD) for all projects
        pd_map = {}
        for _, row in projects_df.iterrows():
            pd_map[row["project_id"]] = self.calculate_default_probability(row["dscr"])

        # Dynamically build mock dependency edges if none provided
        # Connect projects sharing the same country, sector, or sponsors
        if dependency_edges is None:
            edges = []
            for i in range(n_projects):
                p_i = projects_df.iloc[i]
                for j in range(n_projects):
                    if i == j:
                        continue
                    p_j = projects_df.iloc[j]
                    
                    # Dependency criteria
                    weight = 0.0
                    if p_i["country_code"] == p_j["country_code"]:
                        weight += 0.20
                    if p_i["sector"] == p_j["sector"]:
                        weight += 0.15
                    if p_i["sponsors"] == p_j["sponsors"]:
                        weight += 0.25
                        
                    if weight > 0:
                        edges.append({
                            "source": p_j["project_id"],  # project j (affects source)
                            "target": p_i["project_id"],  # project i (target)
                            "weight": weight,
                            "impact": 0.15
                        })
            dependency_edges = pd.DataFrame(edges)

        # Calculate contagion index per project
        contagion_scores = {pid: 0.0 for pid in project_ids}
        
        if not dependency_edges.empty:
            for _, edge in dependency_edges.iterrows():
                src = edge["source"]
                tgt = edge["target"]
                w = edge["weight"]
                imp = edge["impact"]
                
                # Retrieve PD of source project j
                pd_src = pd_map.get(src, 0.05)
                
                if tgt in contagion_scores:
                    contagion_scores[tgt] += w * pd_src * imp

        # Convert to list and round
        results = []
        for pid in project_ids:
            results.append({
                "project_id": pid,
                "default_probability_pd": round(pd_map[pid], 4),
                "portfolio_contagion_index": round(contagion_scores[pid], 4)
            })
            
        return pd.DataFrame(results)

    def fuse_all_features(
        self,
        projects_df: pd.DataFrame,
        satellite_df: pd.DataFrame,
        macro_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fuses financial, satellite, and macroeconomic features, and computes cross-domain features.
        """
        # Ensure index aligns or merge
        merged = pd.merge(projects_df, satellite_df, on="project_id", suffixes=("", "_sat"))
        
        # Merge macro features on country_code
        if "country_code" in merged.columns and "country_code" in macro_df.columns:
            # Macro df might have duplicate country_codes due to timestamps, let's take the latest
            latest_macro = macro_df.sort_values("event_timestamp").groupby("country_code").last().reset_index()
            merged = pd.merge(merged, latest_macro, on="country_code", suffixes=("", "_macro"))
            
        # 1. Compute Construction-Adjusted DSCR
        ca_dscrs = []
        for _, row in merged.iterrows():
            delay = row.get("schedule_delay_months", 0.0)
            # Cost overrun simulated / default 0
            cost_overrun = 0.0
            if "status" in row and row["status"] in ["Distressed", "Cancelled"]:
                cost_overrun = 0.20
            ca_dscr = self.compute_construction_adjusted_dscr(
                base_dscr=row["dscr"],
                schedule_delay_months=delay,
                cost_overrun_pct=cost_overrun
            )
            ca_dscrs.append(ca_dscr)
        merged["construction_adjusted_dscr"] = ca_dscrs

        # 2. Compute Macro-Stress DSCR (using a 2% GDP decline, 150 bps rate increase, 20% FX depreciation stress scenario)
        macro_stressed_dscrs = []
        for _, row in merged.iterrows():
            stressed_dscr = self.compute_macro_stress_dscr(
                base_dscr=row["dscr"],
                sector=row["sector"],
                delta_gdp=-0.02,
                delta_ir=0.015,
                delta_fx=0.20
            )
            macro_stressed_dscrs.append(stressed_dscr)
        merged["macro_stressed_dscr"] = macro_stressed_dscrs

        # 3. Compute Portfolio Contagion Index
        contagion_df = self.compute_portfolio_contagion_index(projects_df)
        merged = pd.merge(merged, contagion_df, on="project_id")
        
        # Drop columns that are not features to return a clean feature dataset
        keep_cols = [
            "project_id",
            "country_code",
            "construction_adjusted_dscr",
            "macro_stressed_dscr",
            "default_probability_pd",
            "portfolio_contagion_index"
        ]
        return merged[keep_cols]

    @staticmethod
    def simulate_monte_carlo_dscr(
        base_dscr: float,
        num_quarters: int = 20,
        num_simulations: int = 100,
        volatility: float = 0.05,
        drift: float = 0.0
    ) -> np.ndarray:
        """
        Simulates Monte Carlo trajectories for DSCR stress analysis.
        Uses a geometric Brownian motion style simulation:
            DSCR_t = DSCR_t-1 * exp((drift - 0.5 * vol^2) + vol * Z_t)
        Returns:
            array of shape (num_simulations, num_quarters + 1)
        """
        trajectories = np.zeros((num_simulations, num_quarters + 1))
        trajectories[:, 0] = base_dscr
        
        for sim in range(num_simulations):
            dscr = base_dscr
            for q in range(1, num_quarters + 1):
                # Stochastic shock
                z = np.random.normal(0, 1)
                # Geometric brownian motion step
                dscr = dscr * np.exp((drift - 0.5 * volatility**2) + volatility * z)
                dscr = max(0.1, dscr)  # Clamping at 0.1
                trajectories[sim, q] = dscr
                
        return trajectories


FEAST_METADATA = {
    "entity": "project_id",
    "features": [
        "construction_adjusted_dscr",
        "macro_stressed_dscr",
        "default_probability_pd",
        "portfolio_contagion_index"
    ]
}

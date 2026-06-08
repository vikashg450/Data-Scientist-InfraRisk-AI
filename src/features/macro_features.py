import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List

class MacroFeaturesCalculator:
    """
    Computes macroeconomic risk indices, sovereign risk composite index,
    governance quality score, and commodity price volatilities.
    """

    def __init__(self, market_data_path: str = "data/market/market_data_combined.csv"):
        self.market_data_path = market_data_path
        self._load_market_data()

    def _load_market_data(self):
        """Loads and caches combined market data for volatility computations and CDS lookups."""
        if os.path.exists(self.market_data_path):
            self.market_df = pd.read_csv(self.market_data_path, parse_dates=["Date"])
            self.market_df.set_index("Date", inplace=True)
            logger_info = f"Loaded market data from {self.market_data_path}"
        else:
            self.market_df = pd.DataFrame()
            logger_info = "Market data file not found, will use offline defaults/simulations."
        print(logger_info)

    @staticmethod
    def map_sovereign_rating_to_score(rating: str) -> float:
        """
        Maps sovereign credit ratings (AAA to D) to a normalized risk score [0, 1].
        0 indicates lowest risk (AAA), 1 indicates highest risk (D).
        """
        rating_map = {
            "AAA": 0.0,
            "AA+": 0.05, "AA": 0.08, "AA-": 0.12,
            "A+": 0.15, "A": 0.18, "A-": 0.22,
            "BBB+": 0.25, "BBB": 0.30, "BBB-": 0.35,
            "BB+": 0.45, "BB": 0.50, "BB-": 0.55,
            "B+": 0.65, "B": 0.70, "B-": 0.75,
            "CCC+": 0.85, "CCC": 0.90, "CCC-": 0.95,
            "CC": 0.98, "C": 0.99, "D": 1.0
        }
        return rating_map.get(rating.strip().upper(), 0.5)

    @staticmethod
    def get_base_rating(rating: str) -> str:
        """Helper to map a rating like BBB- to a base rating category for CDS lookup."""
        rating = rating.strip().upper()
        if rating.startswith("AAA"): return "AAA"
        if rating.startswith("AA"): return "AA"
        if rating.startswith("A"): return "A"
        if rating.startswith("BBB"): return "BBB"
        if rating.startswith("BB"): return "BB"
        if rating.startswith("B"): return "B"
        return "CCC"

    def compute_commodity_volatilities(self, window_days: int = 30) -> Dict[str, float]:
        """
        Calculates the annualized rolling volatilities of key commodities from market data.
        Formula: Vol = Std(Log Returns) * Sqrt(252)
        """
        if self.market_df.empty:
            # Return standard historical defaults if market data is not loaded
            return {
                "crude_oil_volatility_30d": 0.25,
                "natural_gas_volatility_30d": 0.35,
                "steel_volatility_30d": 0.20,
                "cement_volatility_30d": 0.15
            }
        
        vols = {}
        commodities = ["Crude_Oil", "Natural_Gas", "Steel", "Cement_Proxy"]
        for col in commodities:
            if col in self.market_df.columns:
                prices = self.market_df[col]
                # Log returns
                log_ret = np.log(prices / prices.shift(1))
                # Standard deviation
                std_ret = log_ret.rolling(window=window_days).std()
                # Annualized volatility
                ann_vol = std_ret * np.sqrt(252)
                vols[f"{col.lower()}_volatility_30d"] = round(float(ann_vol.iloc[-1]), 4)
            else:
                vols[f"{col.lower()}_volatility_30d"] = 0.20  # Fallback
        return vols

    def compute_governance_composite(self, row: Dict[str, Any]) -> float:
        """
        Aggregates World Bank WGI scores into a composite governance score.
        WGI indicators: regulatory_quality, rule_of_law, control_of_corruption, government_effectiveness
        """
        gov_fields = [
            "regulatory_quality",
            "rule_of_law",
            "control_of_corruption",
            "government_effectiveness"
        ]
        vals = [row[f] for f in gov_fields if f in row and row[f] is not None]
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def compute_fiscal_stress_index(self, row: Dict[str, Any]) -> float:
        """
        Computes a Fiscal Stress Index based on WDI macroeconomic indicators.
        Combines debt/GDP, inflation, and interest rate stability.
        Normalised between 0 (no stress) and 1 (extreme stress).
        """
        rating = row.get("sovereign_rating", "BBB")
        
        # Estimate debt_to_gdp and fiscal_deficit if they are not explicitly present in the data
        # Lower rating countries typically have higher fiscal stress
        if "debt_to_gdp" in row and row["debt_to_gdp"] is not None:
            debt_to_gdp = row["debt_to_gdp"]
        else:
            rating_score = self.map_sovereign_rating_to_score(rating)
            debt_to_gdp = 0.35 + rating_score * 0.50  # Maps BB to ~60%, CCC to ~85%
            
        inflation = row.get("inflation", 4.0)
        real_interest = row.get("real_interest_rate", 2.0)
        
        # Standardize components
        debt_stress = min(1.0, debt_to_gdp / 1.20)  # 120% debt/GDP is extreme stress
        inflation_stress = min(1.0, max(0.0, inflation / 25.0))  # 25% inflation is extreme stress
        interest_stress = min(1.0, max(0.0, real_interest / 15.0))  # 15% real rate is extreme stress
        
        fiscal_stress = (debt_stress * 0.40) + (inflation_stress * 0.30) + (interest_stress * 0.30)
        return round(float(fiscal_stress), 4)

    def compute_sovereign_risk_composite(self, row: Dict[str, Any]) -> float:
        """
        Computes the Composite Sovereign Risk Index by combining sovereign ratings,
        estimated CDS spreads, governance composite, and fiscal stress index.
        Sovereign_Risk_Index = (Rating_Score * 0.4) + (CDS_Score * 0.3) + (Fiscal_Stress * 0.2) + ((1 - Governance) * 0.1)
        """
        rating = row.get("sovereign_rating", "BBB")
        rating_score = self.map_sovereign_rating_to_score(rating)
        
        # Governance score
        gov_comp = self.compute_governance_composite(row)
        # Normalize governance from [-2.5, 2.5] range to [0, 1] where 1 is best, 0 is worst
        gov_comp_norm = np.clip((gov_comp + 2.5) / 5.0, 0.0, 1.0)
        
        # Fiscal stress
        fiscal_stress = self.compute_fiscal_stress_index(row)
        
        # CDS Spread score
        base_rating = self.get_base_rating(rating)
        cds_col = f"CDS_Spread_{base_rating}"
        
        if not self.market_df.empty and cds_col in self.market_df.columns:
            latest_cds = self.market_df[cds_col].iloc[-1]
        else:
            # Mock defaults
            cds_defaults = {
                "AAA": 15.0, "AA": 30.0, "A": 60.0, "BBB": 120.0, "BB": 280.0, "B": 550.0, "CCC": 1200.0
            }
            latest_cds = cds_defaults.get(base_rating, 200.0)
            
        # Normalize CDS between 0 and 1500 bps
        cds_score_norm = min(1.0, latest_cds / 1500.0)
        
        sovereign_risk = (
            (rating_score * 0.40) + 
            (cds_score_norm * 0.30) + 
            (fiscal_stress * 0.20) + 
            ((1.0 - gov_comp_norm) * 0.10)
        )
        return round(float(sovereign_risk), 4)

    def compute_all_macro_features(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes macroeconomic features for a combined projects + WDI DataFrame.
        """
        features_list = []
        comm_vols = self.compute_commodity_volatilities()
        
        for _, row in combined_df.iterrows():
            row_dict = row.to_dict()
            gov_comp = self.compute_governance_composite(row_dict)
            fiscal_stress = self.compute_fiscal_stress_index(row_dict)
            sovereign_risk = self.compute_sovereign_risk_composite(row_dict)
            
            features = {
                "project_id": row["project_id"],
                "country_code": row["country_code"],
                "sovereign_rating_score": self.map_sovereign_rating_to_score(row["sovereign_rating"]),
                "governance_composite_score": round(gov_comp, 4),
                "fiscal_stress_index": round(fiscal_stress, 4),
                "sovereign_risk_composite_index": round(sovereign_risk, 4),
                "event_timestamp": pd.Timestamp.now()
            }
            # Add commodity volatilities
            features.update(comm_vols)
            features_list.append(features)
            
        return pd.DataFrame(features_list)

FEAST_METADATA = {
    "entity": "country_code",
    "features": [
        "sovereign_rating_score",
        "governance_composite_score",
        "fiscal_stress_index",
        "sovereign_risk_composite_index",
        "crude_oil_volatility_30d",
        "natural_gas_volatility_30d",
        "steel_volatility_30d",
        "cement_volatility_30d"
    ]
}

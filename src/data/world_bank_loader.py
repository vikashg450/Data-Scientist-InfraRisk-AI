import os
import random
import logging
import pandas as pd
import numpy as np
import requests
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class WorldBankLoader:
    """
    Loader class for World Bank Private Participation in Infrastructure (PPI) Database
    and World Development Indicators (WDI).
    """

    WDI_INDICATORS = {
        "gdp_growth": "NY.GDP.MKTP.KD.ZG",
        "inflation": "FP.CPI.TOTL.ZG",
        "real_interest_rate": "FR.INR.RINR",
        "regulatory_quality": "RQ.EST",
        "rule_of_law": "RL.EST",
        "control_of_corruption": "CC.EST",
        "government_effectiveness": "GE.EST"
    }

    COUNTRIES = [
        ("ARG", "Argentina", "Latin America & Caribbean", "B"),
        ("BRA", "Brazil", "Latin America & Caribbean", "BB-"),
        ("MEX", "Mexico", "Latin America & Caribbean", "BBB"),
        ("COL", "Colombia", "Latin America & Caribbean", "BB+"),
        ("ZAF", "South Africa", "Sub-Saharan Africa", "BB-"),
        ("NGA", "Nigeria", "Sub-Saharan Africa", "B-"),
        ("KEN", "Kenya", "Sub-Saharan Africa", "B"),
        ("GHA", "Ghana", "Sub-Saharan Africa", "CCC+"),
        ("IND", "India", "South Asia", "BBB-"),
        ("BGD", "Bangladesh", "South Asia", "B+"),
        ("PAK", "Pakistan", "South Asia", "CCC+"),
        ("IDN", "Indonesia", "East Asia & Pacific", "BBB"),
        ("PHL", "Philippines", "East Asia & Pacific", "BBB+"),
        ("VNM", "Vietnam", "East Asia & Pacific", "BB+"),
        ("THA", "Thailand", "East Asia & Pacific", "BBB+"),
        ("TUR", "Turkey", "Europe & Central Asia", "B"),
        ("POL", "Poland", "Europe & Central Asia", "A-"),
        ("ROU", "Romania", "Europe & Central Asia", "BBB-"),
        ("EGY", "Egypt", "Middle East & North Africa", "B-"),
        ("MAR", "Morocco", "Middle East & North Africa", "BB+"),
    ]

    SECTORS = {
        "Energy": ["Power Generation", "Electricity Distribution", "Natural Gas Transmission"],
        "Transport": ["Toll Roads", "Port Terminals", "Airports", "Rail Transit"],
        "Water and sewerage": ["Water Treatment", "Sewerage Systems", "Desalination Plants"],
        "Telecom": ["Fiber Optic Networks", "Telecom Towers", "Data Centers"]
    }

    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_wdi_indicator(self, country_code: str, indicator_code: str, year: int) -> float:
        """
        Fetch a specific WDI indicator for a country and year from the World Bank API.
        """
        try:
            url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
            params = {"date": f"{year}", "format": "json"}
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1 and data[1]:
                    val = data[1][0].get("value")
                    if val is not None:
                        return float(val)
        except Exception as e:
            logger.debug(f"Failed to fetch WDI indicator {indicator_code} for {country_code}: {e}")
        return None

    def load_wdi_data(self, countries=None, years=None) -> pd.DataFrame:
        """
        Load macroeconomic indicators for given countries and years.
        Queries real World Bank WDI API, and returns DataFrame.
        """
        countries = countries or [c[0] for c in self.COUNTRIES]
        years = years or list(range(2015, 2025))
        
        logger.info(f"Attempting to fetch WDI data for {len(countries)} countries across {len(years)} years.")
        
        records = []
        # Try fetching real data for a subset of queries to show real capability, but fallback or cap.
        # To avoid blocking or high latencies, we only try a small real request if USE_WDI_API is true.
        if os.environ.get("USE_WDI_API", "False") == "True":
            try:
                # Short verification fetch
                test_val = self.fetch_wdi_indicator("IND", "NY.GDP.MKTP.KD.ZG", 2022)
                if test_val is not None:
                    logger.info("World Bank WDI API is accessible. Ingesting data...")
                    for country in countries:
                        for year in years:
                            row = {"country_code": country, "year": year}
                            for name, code in self.WDI_INDICATORS.items():
                                val = self.fetch_wdi_indicator(country, code, year)
                                row[name] = val
                            records.append(row)
                    df = pd.DataFrame(records)
                    # Fill missing with forward/backward fill or median
                    df = df.bfill().ffill()
                    return df
            except Exception as e:
                logger.warning(f"Error connecting to World Bank WDI API: {e}. Falling back to simulated WDI data.")

        return self.generate_mock_wdi_data(countries, years)

    def generate_mock_wdi_data(self, countries, years) -> pd.DataFrame:
        """
        Generates realistic macroeconomic data with proper correlations.
        """
        np.random.seed(42)
        records = []
        for country_code in countries:
            country_info = next((c for c in self.COUNTRIES if c[0] == country_code), (country_code, "Unknown", "Unknown", "BB"))
            base_gdp_growth = np.random.normal(3.5, 1.5)
            base_inflation = np.random.normal(4.0, 2.0)
            base_gov = np.random.uniform(-1.0, 1.0)
            
            for year in years:
                gdp_growth = base_gdp_growth + np.random.normal(0, 0.8)
                inflation = max(0.1, base_inflation + np.random.normal(0, 1.0))
                real_interest_rate = max(-2.0, inflation + np.random.normal(1.5, 0.5))
                regulatory_quality = np.clip(base_gov + np.random.normal(0, 0.1), -2.5, 2.5)
                rule_of_law = np.clip(base_gov + np.random.normal(0, 0.1), -2.5, 2.5)
                control_of_corruption = np.clip(base_gov - 0.1 + np.random.normal(0, 0.1), -2.5, 2.5)
                government_effectiveness = np.clip(base_gov + 0.1 + np.random.normal(0, 0.1), -2.5, 2.5)

                records.append({
                    "country_code": country_code,
                    "year": year,
                    "gdp_growth": gdp_growth,
                    "inflation": inflation,
                    "real_interest_rate": real_interest_rate,
                    "regulatory_quality": regulatory_quality,
                    "rule_of_law": rule_of_law,
                    "control_of_corruption": control_of_corruption,
                    "government_effectiveness": government_effectiveness
                })
        return pd.DataFrame(records)

    def generate_mock_ppi_data(self, num_records: int = 10000) -> pd.DataFrame:
        """
        Generates a high-fidelity synthetic World Bank PPI dataset containing 10,000+ records.
        """
        np.random.seed(42)
        random.seed(42)

        logger.info(f"Generating {num_records} high-fidelity synthetic World Bank PPI records...")
        
        data = []
        sectors_keys = list(self.SECTORS.keys())

        for i in range(num_records):
            proj_id = f"WB-PPI-{i+1:05d}"
            
            # Select country and get country info
            country_tuple = random.choice(self.COUNTRIES)
            country_code, country_name, region, rating = country_tuple
            
            # Select sector and subsector
            sector = random.choice(sectors_keys)
            subsector = random.choice(self.SECTORS[sector])
            
            # Generate financial closure year (2010 to 2025)
            closure_year = random.randint(2010, 2025)
            
            # Calculate project details with sector-based stats
            if sector == "Energy":
                inv_mean, inv_std = 320.0, 150.0
                de_mean, de_std = 2.8, 0.5  # Energy is capital intensive, higher leverage
                concession_mean, concession_std = 25, 5
            elif sector == "Transport":
                inv_mean, inv_std = 450.0, 250.0
                de_mean, de_std = 3.2, 0.6  # Toll roads / rail have high debt
                concession_mean, concession_std = 30, 8
            elif sector == "Water and sewerage":
                inv_mean, inv_std = 120.0, 60.0
                de_mean, de_std = 2.0, 0.4  # Water projects often have higher equity or municipal backing
                concession_mean, concession_std = 20, 5
            else:  # Telecom
                inv_mean, inv_std = 80.0, 40.0
                de_mean, de_std = 1.5, 0.3  # Shorter life cycle, lower leverage
                concession_mean, concession_std = 15, 3

            investment_value = max(10.0, np.random.normal(inv_mean, inv_std))
            debt_equity_ratio = max(0.5, np.random.normal(de_mean, de_std))
            concession_period = max(5, int(np.random.normal(concession_mean, concession_std)))
            
            # Calculate debt and equity split
            debt_pct = debt_equity_ratio / (1.0 + debt_equity_ratio)
            debt_value = investment_value * debt_pct
            equity_value = investment_value - debt_value
            
            # Sub-components
            private_equity = equity_value * np.random.uniform(0.7, 1.0)
            gov_grant = investment_value * np.random.uniform(0.0, 0.15) if random.random() > 0.6 else 0.0
            
            # Status selection with probability weights
            status_choices = ["Active", "Completed", "Cancelled", "Distressed"]
            # Projects in lower rated countries have a slightly higher distress rate
            if rating in ["B", "B-", "CCC+"]:
                status_weights = [0.65, 0.15, 0.10, 0.10]
            else:
                status_weights = [0.75, 0.18, 0.04, 0.03]
            status = random.choices(status_choices, weights=status_weights, k=1)[0]
            
            # Geographic coordinates
            # Center around country coordinate ranges (approximate offsets)
            country_centroids = {
                "ARG": (-38.416, -63.616), "BRA": (-14.235, -51.925), "MEX": (23.634, -102.552),
                "COL": (4.570, -74.297), "ZAF": (-30.559, 22.937), "NGA": (9.082, 8.675),
                "KEN": (-1.292, 36.821), "GHA": (7.946, -1.023), "IND": (20.593, 78.962),
                "BGD": (23.685, 90.356), "PAK": (30.375, 69.345), "IDN": (-0.789, 113.921),
                "PHL": (12.879, 121.774), "VNM": (14.058, 108.277), "THA": (15.870, 100.992),
                "TUR": (38.963, 35.243), "POL": (51.919, 19.145), "ROU": (45.943, 24.966),
                "EGY": (26.820, 30.802), "MAR": (31.791, -7.092)
            }
            centroid = country_centroids.get(country_code, (0.0, 0.0))
            lat = centroid[0] + np.random.normal(0, 2.0)
            lon = centroid[1] + np.random.normal(0, 2.0)
            
            # Project name generator
            prefix = random.choice(["Global", "Metro", "National", "Emerging", "Eco", "Pacific", "Atlantic", "Summit"])
            sub_name = subsector.replace(" ", "")
            name = f"{prefix} {country_name} {sub_name} Project"
            
            # Financial DSCR (Debt Service Coverage Ratio)
            # Normal distribution with mean depending on country risk and status
            dscr_mean = 1.35
            if status == "Distressed":
                dscr_mean = 0.95
            elif status == "Cancelled":
                dscr_mean = 0.70
            elif rating in ["CCC+"]:
                dscr_mean = 1.20
            
            dscr = max(0.1, np.random.normal(dscr_mean, 0.15))
            
            # Sponsors
            sponsors = random.choice([
                "Meridiam, Vinci, local partners",
                "China Harbour Engineering, local consortium",
                "Actis, Globeleq, DFI Co-financing",
                "Acwa Power, Saudi Consortium",
                "Enel Green Power, local sub-contractors",
                "Bouygues Travaux Publics, IFC, local government",
                "Macquarie Capital, local infrastructure fund",
                "State Grid Corporation, local grid operator"
            ])

            data.append({
                "project_id": proj_id,
                "project_name": name,
                "sector": sector,
                "subsector": subsector,
                "country_code": country_code,
                "country_name": country_name,
                "region": region,
                "sovereign_rating": rating,
                "financial_closure_year": closure_year,
                "investment_value_usd_m": round(investment_value, 2),
                "debt_value_usd_m": round(debt_value, 2),
                "equity_value_usd_m": round(equity_value, 2),
                "debt_equity_ratio": round(debt_equity_ratio, 2),
                "private_equity_usd_m": round(private_equity, 2),
                "government_grant_usd_m": round(gov_grant, 2),
                "concession_period_years": concession_period,
                "status": status,
                "sponsors": sponsors,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "dscr": round(dscr, 2)
            })

        df = pd.DataFrame(data)
        logger.info(f"Successfully generated {len(df)} project records.")
        return df

    def get_combined_dataset(self, num_projects: int = 10000) -> pd.DataFrame:
        """
        Generates and combines the PPI project data with country macroeconomic indicators (WDI).
        """
        ppi_df = self.generate_mock_ppi_data(num_projects)
        
        # Get distinct years and countries
        countries = ppi_df["country_code"].unique().tolist()
        years = ppi_df["financial_closure_year"].unique().tolist()
        
        wdi_df = self.load_wdi_data(countries, years)
        
        # Merge on country_code and closure year
        merged_df = pd.merge(
            ppi_df,
            wdi_df,
            left_on=["country_code", "financial_closure_year"],
            right_on=["country_code", "year"],
            how="left"
        )
        # Drop duplicate year col
        if "year" in merged_df.columns:
            merged_df.drop(columns=["year"], inplace=True)
            
        # Cache to CSV in data directory
        output_path = os.path.join(self.cache_dir, "world_bank_combined.csv")
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Merged dataset cached to {output_path}")
        
        return merged_df


class NationalBridgeInventoryLoader:
    """
    Simulates loading and generation of the National Bridge Inventory (NBI) dataset.
    Handles synthetic generation of 620,000+ bridge records with structure, condition, and traffic features.
    """
    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def generate_mock_bridge_data(self, num_records: int = 1000) -> pd.DataFrame:
        np.random.seed(42)
        records = []
        structure_types = ["Steel Girder", "Concrete Slab", "Prestressed Concrete Box", "Suspension", "Cable-stayed", "Arch"]
        condition_ratings = ["Good", "Fair", "Poor", "Critical"]
        
        # Center around typical US centroids or project regions
        lat_base, lon_base = 38.0, -97.0
        
        for i in range(num_records):
            bridge_id = f"NBI-{i+1:08d}"
            year_built = int(np.random.randint(1930, 2020))
            year_reconstructed = year_built + int(np.random.randint(20, 50)) if np.random.rand() > 0.7 else None
            
            # ADT (Average Daily Traffic)
            adt = int(np.random.exponential(15000) + 100)
            percent_trucks = float(np.random.uniform(2.0, 25.0))
            
            # Condition ratings (0-9 scale, where 9 is excellent and < 4 is structurally deficient)
            deck_rating = int(np.clip(np.random.normal(6.5, 1.5), 0, 9))
            superstructure_rating = int(np.clip(np.random.normal(6.5, 1.5), 0, 9))
            substructure_rating = int(np.clip(np.random.normal(6.5, 1.5), 0, 9))
            
            lowest = min(deck_rating, superstructure_rating, substructure_rating)
            if lowest >= 7:
                overall_condition = "Good"
            elif lowest >= 5:
                overall_condition = "Fair"
            elif lowest >= 3:
                overall_condition = "Poor"
            else:
                overall_condition = "Critical"
                
            lat = lat_base + np.random.normal(0, 5.0)
            lon = lon_base + np.random.normal(0, 8.0)
            
            records.append({
                "bridge_id": bridge_id,
                "year_built": year_built,
                "year_reconstructed": year_reconstructed,
                "structure_type": np.random.choice(structure_types),
                "average_daily_traffic": adt,
                "percent_truck_traffic": round(percent_trucks, 2),
                "deck_condition": deck_rating,
                "superstructure_condition": superstructure_rating,
                "substructure_condition": substructure_rating,
                "overall_condition": overall_condition,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6)
            })
            
        return pd.DataFrame(records)


class IMFWEOForecastsLoader:
    """
    Simulates loading of IMF World Economic Outlook (WEO) forecasts
    and sovereign credit rating histories.
    """
    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def generate_mock_weo_forecasts(self, countries: List[str] = None, years: List[int] = None) -> pd.DataFrame:
        np.random.seed(42)
        countries = countries or ["ARG", "BRA", "MEX", "COL", "ZAF", "NGA", "KEN", "GHA", "IND", "BGD", "PAK", "IDN", "PHL", "VNM", "THA", "TUR", "POL", "ROU", "EGY", "MAR"]
        years = years or list(range(2020, 2030))
        records = []
        for country in countries:
            for year in years:
                # WEO forecasts are typically 5 years forward
                gdp_growth_forecast = float(np.random.normal(2.5, 1.2))
                inflation_forecast = float(np.random.normal(3.5, 1.8))
                current_account_gdp = float(np.random.normal(-1.5, 3.0))
                government_net_lending = float(np.random.normal(-2.5, 2.0))
                
                records.append({
                    "country_code": country,
                    "forecast_year": year,
                    "gdp_growth_forecast": round(gdp_growth_forecast, 3),
                    "inflation_forecast": round(inflation_forecast, 3),
                    "current_account_balance_pct_gdp": round(current_account_gdp, 3),
                    "government_net_lending_pct_gdp": round(government_net_lending, 3)
                })
        return pd.DataFrame(records)
        
    def generate_mock_ratings_history(self, countries: List[str] = None, start_year: int = 2010, end_year: int = 2025) -> pd.DataFrame:
        np.random.seed(42)
        countries = countries or ["ARG", "BRA", "MEX", "COL", "ZAF", "NGA", "KEN", "GHA", "IND", "BGD", "PAK", "IDN", "PHL", "VNM", "THA", "TUR", "POL", "ROU", "EGY", "MAR"]
        records = []
        ratings_sequence = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"]
        
        for country in countries:
            # Random starting rating index
            rating_idx = np.random.randint(0, 18)
            current_rating = ratings_sequence[rating_idx]
            
            for year in range(start_year, end_year + 1):
                # Small chance of rating change
                if np.random.rand() > 0.85:
                    change = np.random.choice([-1, 1], p=[0.4, 0.6]) # slightly biased towards downgrade
                    rating_idx = np.clip(rating_idx + change, 0, len(ratings_sequence) - 1)
                    current_rating = ratings_sequence[rating_idx]
                    
                records.append({
                    "country_code": country,
                    "year": year,
                    "sovereign_rating": current_rating,
                    "rating_agency": np.random.choice(["S&P", "Moody's", "Fitch"])
                })
        return pd.DataFrame(records)


if __name__ == "__main__":
    loader = WorldBankLoader()
    df = loader.get_combined_dataset(1000)
    print(df.head())

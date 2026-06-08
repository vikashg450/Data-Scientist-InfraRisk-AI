import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates infrastructure and market datasets for completeness,
    range consistency, and physical plausibility.
    """

    CRITICAL_PROJECT_FIELDS = [
        "project_id", "project_name", "sector", "subsector",
        "country_code", "financial_closure_year", "investment_value_usd_m",
        "debt_equity_ratio", "status", "latitude", "longitude", "dscr"
    ]

    VALID_SECTORS = ["Energy", "Transport", "Water and sewerage", "Telecom"]
    VALID_STATUSES = ["Active", "Completed", "Cancelled", "Distressed"]

    def __init__(self, completeness_threshold: float = 0.80):
        self.completeness_threshold = completeness_threshold

    def validate_projects(self, df: pd.DataFrame) -> dict:
        """
        Validates the projects DataFrame.
        Checks completeness rates, coordinate bounds, and financial bounds.
        Flags invalid records and generates a quality report.
        """
        logger.info("Starting validation on projects dataset...")
        
        report = {
            "total_records": len(df),
            "completeness_check": {},
            "range_checks": {},
            "flagged_records_count": 0,
            "overall_status": "PASS"
        }
        
        # 1. Completeness Checks
        # Calculate completeness rate per column
        for col in df.columns:
            comp_rate = df[col].notnull().mean()
            report["completeness_check"][col] = {
                "completeness_rate": float(comp_rate),
                "status": "PASS" if comp_rate >= self.completeness_threshold else "FAIL"
            }
            if comp_rate < self.completeness_threshold:
                report["overall_status"] = "WARNING"

        # Calculate completeness rate per row for critical fields
        critical_df = df[self.CRITICAL_PROJECT_FIELDS]
        row_completeness = critical_df.notnull().mean(axis=1)
        
        # Flag records with completeness < 80%
        flagged_mask = row_completeness < self.completeness_threshold
        flagged_df = df[flagged_mask]
        report["flagged_records_count"] = int(flagged_mask.sum())
        
        if report["flagged_records_count"] > 0:
            logger.warning(f"Flagged {report['flagged_records_count']} records with completeness below {self.completeness_threshold*100}%.")
            # Save flagged records for review
            os.makedirs("data/review", exist_ok=True)
            flagged_df.to_csv("data/review/flagged_projects_review.csv", index=False)
            logger.info("Flagged records saved to data/review/flagged_projects_review.csv")

        # 2. Range & Plausibility Checks
        errors = []

        # Latitude: [-90, 90]
        invalid_lat = df[(df["latitude"] < -90) | (df["latitude"] > 90)]
        report["range_checks"]["latitude"] = {
            "invalid_count": len(invalid_lat),
            "status": "PASS" if len(invalid_lat) == 0 else "FAIL"
        }
        if len(invalid_lat) > 0:
            errors.append(f"Latitude range errors: {len(invalid_lat)} records")

        # Longitude: [-180, 180]
        invalid_lon = df[(df["longitude"] < -180) | (df["longitude"] > 180)]
        report["range_checks"]["longitude"] = {
            "invalid_count": len(invalid_lon),
            "status": "PASS" if len(invalid_lon) == 0 else "FAIL"
        }
        if len(invalid_lon) > 0:
            errors.append(f"Longitude range errors: {len(invalid_lon)} records")

        # DSCR: >= 0
        invalid_dscr = df[df["dscr"] < 0]
        report["range_checks"]["dscr"] = {
            "invalid_count": len(invalid_dscr),
            "status": "PASS" if len(invalid_dscr) == 0 else "FAIL"
        }
        if len(invalid_dscr) > 0:
            errors.append(f"DSCR negative errors: {len(invalid_dscr)} records")

        # Investment Value: > 0
        invalid_inv = df[df["investment_value_usd_m"] <= 0]
        report["range_checks"]["investment_value_usd_m"] = {
            "invalid_count": len(invalid_inv),
            "status": "PASS" if len(invalid_inv) == 0 else "FAIL"
        }
        if len(invalid_inv) > 0:
            errors.append(f"Investment value non-positive errors: {len(invalid_inv)} records")

        # Debt Equity Ratio: >= 0
        invalid_de = df[df["debt_equity_ratio"] < 0]
        report["range_checks"]["debt_equity_ratio"] = {
            "invalid_count": len(invalid_de),
            "status": "PASS" if len(invalid_de) == 0 else "FAIL"
        }
        if len(invalid_de) > 0:
            errors.append(f"Debt-to-equity ratio negative errors: {len(invalid_de)} records")

        # Concession Period: > 0
        invalid_concession = df[df["concession_period_years"] <= 0]
        report["range_checks"]["concession_period_years"] = {
            "invalid_count": len(invalid_concession),
            "status": "PASS" if len(invalid_concession) == 0 else "FAIL"
        }
        if len(invalid_concession) > 0:
            errors.append(f"Concession period non-positive errors: {len(invalid_concession)} records")

        # Sector check
        invalid_sectors = df[~df["sector"].isin(self.VALID_SECTORS)]
        report["range_checks"]["sector"] = {
            "invalid_count": len(invalid_sectors),
            "status": "PASS" if len(invalid_sectors) == 0 else "FAIL"
        }
        if len(invalid_sectors) > 0:
            errors.append(f"Invalid sector names: {len(invalid_sectors)} records")

        # Status check
        invalid_statuses = df[~df["status"].isin(self.VALID_STATUSES)]
        report["range_checks"]["status"] = {
            "invalid_count": len(invalid_statuses),
            "status": "PASS" if len(invalid_statuses) == 0 else "FAIL"
        }
        if len(invalid_statuses) > 0:
            errors.append(f"Invalid status values: {len(invalid_statuses)} records")

        if errors:
            report["overall_status"] = "FAIL"
            report["errors"] = errors
            logger.error(f"Validation failed with the following errors: {errors}")
        else:
            logger.info("Validation checks passed successfully.")
            
        return report

    def validate_market_data(self, df: pd.DataFrame) -> dict:
        """
        Validates the market data.
        Checks for completeness and logical commodity and yield constraints.
        """
        logger.info("Starting validation on market dataset...")
        report = {
            "total_records": len(df),
            "completeness_check": {},
            "range_checks": {},
            "overall_status": "PASS"
        }
        
        # Completeness
        for col in df.columns:
            comp_rate = df[col].notnull().mean()
            report["completeness_check"][col] = {
                "completeness_rate": float(comp_rate),
                "status": "PASS" if comp_rate >= self.completeness_threshold else "FAIL"
            }
            if comp_rate < self.completeness_threshold:
                report["overall_status"] = "WARNING"
                
        # Range checks
        errors = []
        
        # Commodities (Crude oil, gas, steel, cement proxy should be positive)
        for col in ["Crude_Oil", "Natural_Gas", "Steel", "Cement_Proxy"]:
            if col in df.columns:
                invalid_vals = df[df[col] <= 0]
                report["range_checks"][col] = {
                    "invalid_count": len(invalid_vals),
                    "status": "PASS" if len(invalid_vals) == 0 else "FAIL"
                }
                if len(invalid_vals) > 0:
                    errors.append(f"Market commodity '{col}' has non-positive prices: {len(invalid_vals)} rows")

        # US Yields (Should be bounded, say between -2% and 15%)
        for name in ["3M", "5Y", "10Y", "30Y"]:
            colname = f"US_Yield_{name}"
            if colname in df.columns:
                invalid_vals = df[(df[colname] < -2.0) | (df[colname] > 15.0)]
                report["range_checks"][colname] = {
                    "invalid_count": len(invalid_vals),
                    "status": "PASS" if len(invalid_vals) == 0 else "FAIL"
                }
                if len(invalid_vals) > 0:
                    errors.append(f"Interest rate yield '{colname}' falls outside logical range [-2%, 15%]: {len(invalid_vals)} rows")
                    
        if errors:
            report["overall_status"] = "FAIL"
            report["errors"] = errors
            logger.error(f"Market validation failed: {errors}")
        else:
            logger.info("Market validation checks passed successfully.")
            
        return report

    def create_great_expectations_suite(self, df: pd.DataFrame, suite_name: str = "project_suite") -> bool:
        """
        Saves a programmatically configured Great Expectations validation suite.
        If Great Expectations is not installed, it catches the ImportException gracefully.
        """
        try:
            import great_expectations as ge
            # Convert pandas df to Great Expectations dataset
            ge_df = ge.from_pandas(df)
            
            # Add expectations
            ge_df.expect_column_values_to_not_be_null("project_id")
            ge_df.expect_column_values_to_be_in_set("sector", self.VALID_SECTORS)
            ge_df.expect_column_values_to_be_in_set("status", self.VALID_STATUSES)
            ge_df.expect_column_values_to_be_between("latitude", -90, 90)
            ge_df.expect_column_values_to_be_between("longitude", -180, 180)
            ge_df.expect_column_values_to_be_between("dscr", 0, None)
            ge_df.expect_column_values_to_be_between("investment_value_usd_m", 0, None)
            ge_df.expect_column_values_to_be_between("debt_equity_ratio", 0, None)
            ge_df.expect_column_values_to_be_between("concession_period_years", 0, None)
            
            # Save suite
            suite = ge_df.get_expectation_suite()
            os.makedirs("configs/great_expectations", exist_ok=True)
            import json
            with open(f"configs/great_expectations/{suite_name}.json", "w") as f:
                json.dump(suite.to_json_dict(), f, indent=4)
            logger.info(f"Programmatic Great Expectations suite saved to configs/great_expectations/{suite_name}.json")
            return True
        except ImportError:
            logger.warning("great_expectations is not installed. Skipping Great Expectations suite serialization.")
            return False

if __name__ == "__main__":
    from world_bank_loader import WorldBankLoader
    loader = WorldBankLoader()
    df = loader.get_combined_dataset(100)
    
    validator = DataValidator()
    report = validator.validate_projects(df)
    print("Validation Report overall status:", report["overall_status"])
    validator.create_great_expectations_suite(df)

import os
import pytest
import pandas as pd
import numpy as np
import rasterio
from src.data.world_bank_loader import WorldBankLoader, NationalBridgeInventoryLoader, IMFWEOForecastsLoader
from src.data.satellite_downloader import SatelliteDownloader
from src.data.market_data_loader import MarketDataLoader
from src.data.data_validator import DataValidator

def test_world_bank_loader_mock():
    """
    Test that the World Bank mock generator creates the correct structure and number of records.
    """
    loader = WorldBankLoader(cache_dir="tests/test_data")
    # Generate a smaller set of projects to speed up tests
    df = loader.generate_mock_ppi_data(num_records=500)
    
    assert len(df) == 500
    assert "project_id" in df.columns
    assert "project_name" in df.columns
    assert "sector" in df.columns
    assert "investment_value_usd_m" in df.columns
    assert "dscr" in df.columns
    assert "latitude" in df.columns
    assert "longitude" in df.columns
    
    # Check that sectors are valid
    assert df["sector"].isin(loader.SECTORS.keys()).all()
    # Check that statuses are valid
    assert df["status"].isin(["Active", "Completed", "Cancelled", "Distressed"]).all()

def test_satellite_downloader_mock():
    """
    Test that the satellite downloader mock raster generator produces a valid 13-band GeoTIFF.
    """
    downloader = SatelliteDownloader(data_dir="tests/test_data/satellite")
    path = downloader.generate_mock_raster(lat=10.0, lon=20.0, project_id="TEST-001", width=64, height=64)
    
    assert os.path.exists(path)
    
    with rasterio.open(path) as src:
        assert src.count == 13
        assert src.width == 64
        assert src.height == 64
        assert src.crs is not None
        assert src.transform is not None
        
        # Read B8 (index 8) and B4 (index 4) to verify we can do NDVI calculations
        b8 = src.read(8)
        b4 = src.read(4)
        assert b8.shape == (64, 64)
        assert b4.shape == (64, 64)
        
    # Clean up test file
    if os.path.exists(path):
        os.remove(path)

def test_market_data_loader_synthetic():
    """
    Test that the market data loader generates synthetic time series correctly.
    """
    loader = MarketDataLoader(cache_dir="tests/test_data/market")
    df = loader.generate_synthetic_market_data(start_date="2023-01-01", end_date="2023-03-01")
    
    assert not df.empty
    # Check critical columns exist
    assert "Crude_Oil" in df.columns
    assert "Natural_Gas" in df.columns
    assert "Steel" in df.columns
    assert "Cement_Proxy" in df.columns
    assert "USD_EUR" in df.columns
    assert "US_Yield_10Y" in df.columns
    assert "sofr_ns_beta0" in df.columns
    assert "euribor_ns_beta0" in df.columns
    assert "CDS_Spread_BB" in df.columns

def test_data_validator():
    """
    Test that the data validator flags invalid records and reports completeness.
    """
    # Create sample dataframe with some invalid rows
    data = [
        {
            "project_id": "P1", "project_name": "Valid 1", "sector": "Energy", "subsector": "Solar",
            "country_code": "IND", "financial_closure_year": 2020, "investment_value_usd_m": 150.0,
            "debt_equity_ratio": 2.0, "status": "Active", "latitude": 20.0, "longitude": 78.0, "dscr": 1.25,
            "concession_period_years": 25
        },
        {
            # Invalid latitude, negative DSCR, invalid status
            "project_id": "P2", "project_name": "Invalid 1", "sector": "Transport", "subsector": "Road",
            "country_code": "BRA", "financial_closure_year": 2021, "investment_value_usd_m": 200.0,
            "debt_equity_ratio": 3.0, "status": "UnknownStatus", "latitude": 150.0, "longitude": -45.0, "dscr": -0.5,
            "concession_period_years": 30
        },
        {
            # Highly incomplete row
            "project_id": "P3", "project_name": "Incomplete", "sector": "Energy", "subsector": "Wind",
            "country_code": "ARG", "financial_closure_year": 2022, "investment_value_usd_m": None,
            "debt_equity_ratio": None, "status": "Active", "latitude": -34.0, "longitude": -58.0, "dscr": None,
            "concession_period_years": None
        }
    ]
    df = pd.DataFrame(data)
    
    validator = DataValidator(completeness_threshold=0.80)
    report = validator.validate_projects(df)
    
    # P3 has 4 null values out of 12 fields, so completeness is 8/12 = 66.6% < 80%. It should be flagged.
    assert report["flagged_records_count"] == 1
    
    # We should have failure statuses for latitude, dscr, status, sector/status checks
    assert report["range_checks"]["latitude"]["status"] == "FAIL"
    assert report["range_checks"]["dscr"]["status"] == "FAIL"
    assert report["range_checks"]["status"]["status"] == "FAIL"
    
    # Clean up flagged reviews if created
    review_path = "data/review/flagged_projects_review.csv"
    if os.path.exists(review_path):
        os.remove(review_path)

def test_bridge_and_imf_loaders():
    """
    Test the National Bridge Inventory and IMF WEO/Ratings loaders.
    """
    bridge_loader = NationalBridgeInventoryLoader(cache_dir="tests/test_data")
    df_bridge = bridge_loader.generate_mock_bridge_data(num_records=100)
    
    assert len(df_bridge) == 100
    assert "bridge_id" in df_bridge.columns
    assert "overall_condition" in df_bridge.columns
    assert df_bridge["overall_condition"].isin(["Good", "Fair", "Poor", "Critical"]).all()
    
    imf_loader = IMFWEOForecastsLoader(cache_dir="tests/test_data")
    df_forecasts = imf_loader.generate_mock_weo_forecasts(countries=["KEN", "IND"], years=[2023, 2024])
    df_ratings = imf_loader.generate_mock_ratings_history(countries=["KEN", "IND"], start_year=2020, end_year=2022)
    
    assert len(df_forecasts) == 4
    assert "gdp_growth_forecast" in df_forecasts.columns
    assert len(df_ratings) == 6
    assert "sovereign_rating" in df_ratings.columns

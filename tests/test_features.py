import os
import pytest
import numpy as np
import pandas as pd
import rasterio
from src.features.financial_features import FinancialFeaturesCalculator
from src.features.satellite_features import SatelliteFeaturesExtractor
from src.features.macro_features import MacroFeaturesCalculator
from src.features.fusion_features import FusionFeaturesCalculator
from src.features.nlp_features import NLPContractFeaturesExtractor
from src.data.satellite_downloader import SatelliteDownloader
from src.data.market_data_loader import MarketDataLoader

# ----------------- Financial Features Tests -----------------

def test_dscr_calculation():
    calc = FinancialFeaturesCalculator()
    assert calc.calculate_dscr(120.0, 100.0) == 1.2
    assert calc.calculate_dscr(50.0, 0.0) == 99.0
    assert calc.calculate_dscr(-10.0, 100.0) == 0.0

def test_llcr_and_plcr_calculations():
    calc = FinancialFeaturesCalculator()
    cfads_series = [100.0, 105.0, 110.0, 115.0]
    out_debt = 350.0
    interest_rate = 0.05
    
    # LLCR
    llcr = calc.calculate_llcr(cfads_series, out_debt, interest_rate)
    # Expected NPV = 100/(1.05)^1 + 105/(1.05)^2 + 110/(1.05)^3 + 115/(1.05)^4
    expected_npv = (
        100 / 1.05 + 
        105 / (1.05**2) + 
        110 / (1.05**3) + 
        115 / (1.05**4)
    )
    expected_llcr = expected_npv / out_debt
    assert pytest.approx(llcr, 0.001) == expected_llcr
    
    # PLCR
    plcr = calc.calculate_plcr(cfads_series, out_debt, interest_rate)
    assert pytest.approx(plcr, 0.001) == expected_llcr
    
    # Boundary / Zero debt cases
    assert calc.calculate_llcr(cfads_series, 0.0, interest_rate) == 99.0
    assert calc.calculate_plcr(cfads_series, 0.0, interest_rate) == 99.0

def test_spv_waterfall_simulation():
    calc = FinancialFeaturesCalculator()
    waterfall = calc.simulate_spv_waterfall(
        investment_value=100.0,
        debt_value=70.0,
        concession_period=20,
        base_dscr=1.35,
        sector="Energy",
        interest_rate=0.05,
        sweep_pct=0.50,
        construction_period=3
    )
    
    assert len(waterfall) == 20
    # First 3 years should be construction phase
    for yr in range(3):
        assert waterfall[yr]["phase"] == "construction"
        assert waterfall[yr]["revenue"] == 0.0
        
    # Later years should be operations phase
    for yr in range(3, 20):
        assert waterfall[yr]["phase"] == "operations"
        # EBITDA should be positive and DSCR should be around the target or positive
        assert waterfall[yr]["ebitda"] > 0.0
        assert waterfall[yr]["dscr"] >= 0.0
        assert waterfall[yr]["llcr"] >= 0.0
        assert waterfall[yr]["plcr"] >= 0.0

def test_project_features_extraction():
    calc = FinancialFeaturesCalculator()
    # Create mock projects dataframe
    df = pd.DataFrame([
        {
            "project_id": "P-TEST-1",
            "investment_value_usd_m": 120.0,
            "debt_value_usd_m": 80.0,
            "debt_equity_ratio": 2.0,
            "concession_period_years": 25,
            "dscr": 1.40,
            "sector": "Transport"
        }
    ])
    features_df = calc.compute_project_features(df)
    assert not features_df.empty
    assert "leverage_ratio" in features_df.columns
    assert "simulated_avg_dscr" in features_df.columns
    assert features_df.iloc[0]["leverage_ratio"] == pytest.approx(80.0/120.0, 0.01)

# ----------------- Satellite Features Tests -----------------

def test_satellite_features_indices():
    # Setup mock raster using SatelliteDownloader
    downloader = SatelliteDownloader(data_dir="tests/test_data/satellite")
    path = downloader.generate_mock_raster(lat=20.0, lon=78.0, project_id="P-SAT-TEST", width=32, height=32)
    
    extractor = SatelliteFeaturesExtractor(data_dir="tests/test_data/satellite")
    bands = extractor.load_raster_bands(path)
    
    assert "B4" in bands
    assert "B8" in bands
    assert "B12" in bands
    
    ndvi = extractor.calculate_ndvi(bands)
    ndbi = extractor.calculate_ndbi(bands)
    
    assert ndvi.shape == (32, 32)
    assert ndbi.shape == (32, 32)
    
    # Calculate raster features
    feats = extractor.extract_raster_features(path)
    assert "mean_ndvi" in feats
    assert "mean_ndbi" in feats
    
    # Cleanup
    if os.path.exists(path):
        os.remove(path)

def test_satellite_temporal_curves():
    extractor = SatelliteFeaturesExtractor(data_dir="tests/test_data/satellite")
    
    # Test S-curve progress curve
    p_early = extractor.compute_progress_curve(5, 36)
    p_mid = extractor.compute_progress_curve(18, 36)
    p_late = extractor.compute_progress_curve(31, 36)
    
    assert 0.0 <= p_early <= p_mid <= p_late <= 1.0
    
    # Test project satellite features mock fallback
    feats = extractor.extract_project_satellite_features(
        project_id="P-NON-EXISTENT",
        elapsed_months=12,
        planned_duration_months=36
    )
    
    assert "mean_ndvi" in feats
    assert "mean_ndbi" in feats
    assert "satellite_progress_estimate" in feats
    assert "schedule_delay_months" in feats
    
# ----------------- Macro Features Tests -----------------

def test_macro_features_scores():
    # Test rating mapping
    assert MacroFeaturesCalculator.map_sovereign_rating_to_score("AAA") == 0.0
    assert MacroFeaturesCalculator.map_sovereign_rating_to_score("BBB") == 0.30
    assert MacroFeaturesCalculator.map_sovereign_rating_to_score("CCC") == 0.90
    
    # Setup calculator (without market data file, using defaults)
    calc = MacroFeaturesCalculator(market_data_path="non_existent_file.csv")
    
    # Test composite governance
    row = {
        "regulatory_quality": 0.5,
        "rule_of_law": 0.4,
        "control_of_corruption": -0.2,
        "government_effectiveness": 0.3
    }
    gov = calc.compute_governance_composite(row)
    assert gov == np.mean([0.5, 0.4, -0.2, 0.3])
    
    # Test fiscal stress
    stress = calc.compute_fiscal_stress_index({
        "sovereign_rating": "BBB",
        "inflation": 5.0,
        "real_interest_rate": 3.0
    })
    assert 0.0 <= stress <= 1.0
    
    # Test sovereign risk index
    sov_risk = calc.compute_sovereign_risk_composite({
        "sovereign_rating": "BB",
        "regulatory_quality": 0.1,
        "rule_of_law": 0.0,
        "control_of_corruption": -0.2,
        "government_effectiveness": 0.1,
        "inflation": 6.0,
        "real_interest_rate": 2.5
    })
    assert 0.0 <= sov_risk <= 1.0

# ----------------- Fusion Features Tests -----------------

def test_fusion_features():
    # Test Construction-Adjusted DSCR
    # CA_DSCR = 1.40 * (1 - 6 * 0.01333 - 0.10 * 0.5) = 1.40 * (1 - 0.08 - 0.05) = 1.40 * 0.87 = 1.218
    ca_dscr = FusionFeaturesCalculator.compute_construction_adjusted_dscr(
        base_dscr=1.40,
        schedule_delay_months=6.0,
        cost_overrun_pct=0.10
    )
    assert pytest.approx(ca_dscr, 0.01) == 1.218

    # Test Macro-Stress DSCR
    stressed = FusionFeaturesCalculator().compute_macro_stress_dscr(
        base_dscr=1.40,
        sector="Transport",
        delta_gdp=-0.02,
        delta_ir=0.015,
        delta_fx=0.20
    )
    # Sensitivities for Transport: gdp=1.2, ir=0.2, fx=0.3
    # Stressed = 1.40 * (1 + 1.2 * -0.02) * (1 - 0.2 * 0.015) * (1 - 0.3 * 0.20)
    # Stressed = 1.40 * (1 - 0.024) * (1 - 0.003) * (1 - 0.06)
    # Stressed = 1.40 * 0.976 * 0.997 * 0.940 = 1.2807
    assert pytest.approx(stressed, 0.01) == 1.281

    # Test Contagion Index
    # Setup mock projects df
    projects_df = pd.DataFrame([
        {"project_id": "P1", "dscr": 1.40, "country_code": "IND", "sector": "Transport", "sponsors": "S1"},
        {"project_id": "P2", "dscr": 1.05, "country_code": "IND", "sector": "Transport", "sponsors": "S1"},
        {"project_id": "P3", "dscr": 1.60, "country_code": "BRA", "sector": "Energy", "sponsors": "S2"}
    ])
    
    calc = FusionFeaturesCalculator()
    contagion_df = calc.compute_portfolio_contagion_index(projects_df)
    
    assert len(contagion_df) == 3
    assert "portfolio_contagion_index" in contagion_df.columns
    # P2 is higher risk (dscr=1.05), so it should cause contagion to P1 (which shares country/sector/sponsor)
    p1_contagion = contagion_df.loc[contagion_df["project_id"] == "P1", "portfolio_contagion_index"].values[0]
    p3_contagion = contagion_df.loc[contagion_df["project_id"] == "P3", "portfolio_contagion_index"].values[0]
    
    # P1 has shared dependencies with P2. P3 has none with P2 (different country, sector, sponsor).
    assert p1_contagion > p3_contagion
    
    # Test Monte Carlo simulation
    trajectories = FusionFeaturesCalculator.simulate_monte_carlo_dscr(base_dscr=1.40, num_quarters=8, num_simulations=50)
    assert trajectories.shape == (50, 9)
    assert np.all(trajectories[:, 0] == 1.40)
    assert np.all(trajectories >= 0.1)

# ----------------- NLP Features Tests -----------------

def test_nlp_features():
    extractor = NLPContractFeaturesExtractor()
    
    text = """
    This Agreement is signed between Kenya Roads Authority (the Sponsor) and Nairobi Highway Ltd (the Contractor).
    Value is USD 145,000,000. In case of natural disaster or act of god, force majeure is declared.
    """
    
    feats = extractor.extract_features_from_text("P-NLP-TEST", text)
    assert feats["project_id"] == "P-NLP-TEST"
    assert feats["num_parties"] > 0
    assert feats["has_force_majeure"] == 1
    assert feats["contract_risk_score"] > 0.0
    
    # Test batch processing
    contracts = [
        {"project_id": "P-NLP-1", "contract_text": text},
        {"project_id": "P-NLP-2", "contract_text": "Governing law is courts of Kenya."}
    ]
    df = extractor.compute_all_nlp_features(contracts)
    assert len(df) == 2
    assert "contract_risk_score" in df.columns
    assert "has_governing_law" in df.columns

if __name__ == "__main__":
    print("Running financial features tests...")
    test_dscr_calculation()
    test_llcr_and_plcr_calculations()
    test_spv_waterfall_simulation()
    test_project_features_extraction()
    print("Financial features tests passed!")
    
    print("Running satellite features tests...")
    test_satellite_features_indices()
    test_satellite_temporal_curves()
    print("Satellite features tests passed!")
    
    print("Running macro features tests...")
    test_macro_features_scores()
    print("Macro features tests passed!")
    
    print("Running fusion features tests...")
    test_fusion_features()
    print("Fusion features tests passed!")
    
    print("Running NLP features tests...")
    test_nlp_features()
    print("NLP features tests passed!")
    
    print("All tests passed successfully!")


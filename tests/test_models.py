import os
import pytest
import numpy as np
import pandas as pd
import torch
from src.models.cnn.satellite_cnn import SatelliteSiameseCNN
from src.models.tft.demand_forecaster import (
    TFTForecaster, SARIMABaseline, SectorDemandForecaster,
    TollRoadDemandModel, PowerPlantDemandModel, PortDemandModel, AirportDemandModel,
    calculate_picp, calculate_crps, calculate_winkler_score
)
from src.models.gnn.portfolio_gnn import PortfolioGNN
from src.models.pinn.degradation_pinn import DegradationPINN
from src.models.nlp.contract_nlp import ContractNLPAnalyzer
from src.models.xgb.credit_scorer import CreditScorerXGB, CreditScorerLGBM
from src.models.ensemble.stacking_ensemble import StackingEnsembleMetaLearner
from src.data.satellite_downloader import SatelliteDownloader
from src.features.fusion_features import FusionFeaturesCalculator
import rasterio

# ----------------- CNN Model Tests -----------------
def test_satellite_cnn():
    model = SatelliteSiameseCNN(num_classes=5, in_channels=13)
    
    # Mock inputs (batch size 2, 13 bands, 64x64 resolution)
    img_before = torch.randn(2, 13, 64, 64)
    img_current = torch.randn(2, 13, 64, 64)
    
    progress, phase_logits, anomaly_logits = model(img_before, img_current)
    
    assert progress.shape == (2, 1)
    assert phase_logits.shape == (2, 5)
    assert anomaly_logits.shape == (2, 2)
    
    # Save/load model
    weights_path = "tests/test_data/satellite_cnn_test.pth"
    # Ensure dir exists
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    model.save_model(weights_path)
    assert os.path.exists(weights_path)
    
    new_model = SatelliteSiameseCNN(num_classes=5, in_channels=13)
    new_model.load_model(weights_path)
    
    # Clean up
    if os.path.exists(weights_path):
        os.remove(weights_path)

# ----------------- TFT & SARIMA Model Tests -----------------
def test_demand_forecasters():
    # TFT tests
    forecaster = TFTForecaster(num_static=4, num_past=10, num_future=5, d_model=16, num_heads=2)
    
    # Mock datasets
    N = 100
    x_static = np.random.randn(N, 4)
    x_past = np.random.randn(N, 12, 10)
    x_future = np.random.randn(N, 6, 5)
    y_future = np.random.randn(N, 6, 1)
    
    # Fit and Predict
    forecaster.fit(x_static, x_past, x_future, y_future, epochs=2, batch_size=16)
    p10, p50, p90 = forecaster.predict(x_static, x_past, x_future)
    
    assert p10.shape == (N, 6)
    assert p50.shape == (N, 6)
    assert p90.shape == (N, 6)
    
    # SARIMA tests
    sarima = SARIMABaseline(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))
    y_series = pd.Series(np.sin(np.linspace(0, 10, 50)) + np.random.normal(0, 0.1, 50))
    sarima.fit(y_series)
    p10_s, p50_s, p90_s = sarima.predict(steps=5)
    
    assert len(p10_s) == 5
    assert len(p50_s) == 5
    assert len(p90_s) == 5

    # Sector-specific demand forecaster tests
    sector_forecaster = SectorDemandForecaster(sector="toll_roads", num_static=4, num_past=10, num_future=5)
    
    # We will slice mock datasets to match the sizes for dynamic dimensions
    x_static_sect = np.random.randn(50, 4)
    x_past_sect = np.random.randn(50, 12, 10)
    x_future_sect = np.random.randn(50, 6, 5)
    y_future_sect = np.random.randn(50, 6, 1)
    y_series_sect = pd.Series(np.sin(np.linspace(0, 10, 50)) + np.random.normal(0, 0.1, 50))
    
    sector_forecaster.fit(x_static_sect, x_past_sect, x_future_sect, y_future_sect, y_series_sect)
    preds = sector_forecaster.predict(x_static_sect, x_past_sect, x_future_sect, steps=6)
    
    assert "TFT" in preds
    assert "SARIMA" in preds
    assert preds["TFT"][0].shape == (50, 6)
    assert len(preds["SARIMA"][0]) == 6

# ----------------- GNN Model Tests -----------------
def test_portfolio_gnn():
    model = PortfolioGNN(in_features=8, hidden_dim=16, out_dim=8)
    
    # Mock graph with 4 nodes, 8 features each
    x = torch.randn(4, 8)
    # Directed edge list: 0->1, 1->2, 2->3, 3->0
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_weight = torch.tensor([0.5, 0.8, 0.2, 0.9], dtype=torch.float32)
    
    embeddings, pd_scores = model(x, edge_index, edge_weight)
    
    assert embeddings.shape == (4, 8)
    assert pd_scores.shape == (4, 1)
    
    # Contagion propagation
    initial_pd = np.array([0.05, 0.10, 0.02, 0.01])
    edge_index_np = edge_index.numpy()
    edge_weight_np = edge_weight.numpy()
    
    updated_pd = PortfolioGNN.compute_contagion_index(
        initial_pd=initial_pd,
        edge_index=edge_index_np,
        edge_weight=edge_weight_np,
        num_nodes=4,
        alpha=0.5,
        steps=3
    )
    
    assert len(updated_pd) == 4
    assert np.all(updated_pd >= 0.0) & np.all(updated_pd <= 1.0)

# ----------------- PINN Model Tests -----------------
def test_degradation_pinn():
    pinn = DegradationPINN(hidden_dim=16)
    
    # Test forward pass
    t_test = torch.linspace(0.0, 10.0, 10).view(-1, 1).requires_grad_(True)
    a, P, d_c = pinn(t_test)
    
    assert a.shape == (10, 1)
    assert P.shape == (10, 1)
    assert d_c.shape == (10, 1)
    
    # Test residuals
    physics_params = {
        "C": 1.2e-11, 
        "m": 3.0, 
        "Y": 1.12, 
        "d_sigma": 150.0, 
        "f_cycles": 1e5, 
        "beta": 0.05, 
        "alpha": 0.8, 
        "corr_A": 0.002, 
        "corr_B": 0.5
    }
    res_paris, res_aashto, res_corrosion = pinn.compute_physics_residuals(t_test, a, P, d_c, physics_params)
    
    assert res_paris.shape == (10, 1)
    assert res_aashto.shape == (10, 1)
    assert res_corrosion.shape == (10, 1)
    
    # Test training fit
    t_data = np.linspace(0.1, 10.0, 50)
    a_data = 0.001 + 0.002 * t_data
    P_data = 4.5 - 0.05 * t_data
    dc_data = 0.001 * np.sqrt(t_data)
    
    pinn.fit(t_data, a_data, P_data, dc_data, physics_params, epochs=5, lr=1e-3, lambda_phys=0.01)
    
    # Test RUL prediction
    rul, factor = pinn.predict_rul(current_t=2.0, physics_params=physics_params, max_years=20.0)
    assert rul >= 0.0
    assert isinstance(factor, str)
    
    # Test Climate-Adjusted RUL prediction
    rul_rcp85, factor_rcp85 = pinn.predict_climate_adjusted_rul(current_t=2.0, physics_params=physics_params, scenario="RCP8.5", max_years=20.0)
    assert rul_rcp85 >= 0.0
    assert "RCP8.5" in factor_rcp85 or "No limit" in factor_rcp85
    
    rul_rcp45, factor_rcp45 = pinn.predict_climate_adjusted_rul(current_t=2.0, physics_params=physics_params, scenario="RCP4.5", max_years=20.0)
    assert rul_rcp45 >= 0.0
    assert "RCP4.5" in factor_rcp45 or "No limit" in factor_rcp45


# ----------------- NLP Pipeline Tests -----------------
def test_contract_nlp():
    analyzer = ContractNLPAnalyzer()
    
    text = """
    This CONTRACT is signed on 2024-05-15 by Nairobi Highway Authority (the Employer) and Kenya Roads Ltd. (the Contractor).
    The total project value is USD 145,000,000.
    In the event of an act of god, extreme natural disaster, war, or epidemic, a force majeure event shall be declared.
    Either party may terminate the contract upon ninety days written notice if there is a material adverse change in financial condition.
    """
    
    # Test entities extraction
    entities = analyzer.extract_entities(text)
    assert any("Nairobi" in party or "Kenya" in party for party in entities["parties"])
    assert "2024-05-15" in entities["dates"]
    assert "USD 145,000,000" in entities["amounts"] or "$ 145,000,000" in entities["amounts"] or "145,000,000" in entities["amounts"][0]
    
    # Test clause classification
    probs = analyzer.classify_clause("In the event of an act of god, storm or earthquake, force majeure applies.")
    assert probs["force_majeure"] > probs["governing_law"]
    
    # Test risk report
    report = analyzer.generate_risk_report(text)
    assert "overall_risk_score" in report
    assert len(report["entities"]["parties"]) > 0

# ----------------- XGBoost & LightGBM Tests -----------------
def test_credit_scorer():
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"feat_{i}" for i in range(5)])
    y = np.random.randint(0, 2, 100)
    
    # XGB Scorer
    xgb_scorer = CreditScorerXGB()
    xgb_scorer.fit(X, y)
    probs_xgb = xgb_scorer.predict_proba(X)
    assert len(probs_xgb) == 100
    
    eval_xgb = xgb_scorer.evaluate(X, y)
    assert "auroc" in eval_xgb
    
    # LightGBM Scorer
    lgb_scorer = CreditScorerLGBM()
    lgb_scorer.fit(X, y)
    probs_lgb = lgb_scorer.predict_proba(X)
    assert len(probs_lgb) == 100
    
    eval_lgb = lgb_scorer.evaluate(X, y)
    assert "auroc" in eval_lgb

# ----------------- Stacking Ensemble Tests -----------------
def test_stacking_ensemble():
    learner = StackingEnsembleMetaLearner()
    
    # Mock base predictions
    gnn_pd = np.random.uniform(0.01, 0.15, 50)
    tft_dist = np.random.uniform(0.02, 0.20, 50)
    xgb_pd = np.random.uniform(0.01, 0.10, 50)
    
    meta_features = learner.construct_meta_features(gnn_pd, tft_dist, xgb_pd)
    assert meta_features.shape == (50, 3)
    
    # Fit and evaluate
    y = np.random.randint(0, 2, 50)
    learner.fit(meta_features, y)
    
    final_pd = learner.predict_proba(meta_features)
    assert len(final_pd) == 50
    assert np.all(final_pd >= 0.0) & np.all(final_pd <= 1.0)
    
    # Expected Loss
    el = StackingEnsembleMetaLearner.compute_expected_loss(pd_value=0.05, lgd_value=0.45, ead_value=120.0)
    assert el == 0.05 * 0.45 * 120.0

# ----------------- Satellite Preprocessor Tests -----------------
def test_satellite_preprocessors():
    downloader = SatelliteDownloader(data_dir="tests/test_data/satellite_prep")
    path_before = downloader.generate_mock_raster(lat=10.0, lon=20.0, project_id="PREP_BEFORE", width=32, height=32)
    path_current = downloader.generate_mock_raster(lat=10.0, lon=20.0, project_id="PREP_CURRENT", width=32, height=32)
    
    # 1. Atmospheric correction
    corrected = downloader.apply_atmospheric_correction(path_current)
    assert os.path.exists(corrected)
    with rasterio.open(corrected) as src:
        assert src.tags().get("processing_level") == "Level-2A (BOA)"
        
    # 2. Cloud masking
    masked, pct = downloader.apply_cloud_masking_fmask(path_current)
    assert 0.0 <= pct <= 45.0
    with rasterio.open(masked) as src:
        assert src.tags().get("cloud_masked") == "True"
        
    # 3. SIFT Co-registration
    aligned_before, aligned_current = downloader.apply_sift_co_registration(path_before, path_current)
    assert os.path.exists(aligned_before) and os.path.exists(aligned_current)
    
    # 4. Radiometric normalization
    norm = downloader.apply_radiometric_normalisation(path_current, path_before)
    assert os.path.exists(norm)
    
    # Clean up
    for p in [path_before, path_current]:
        if os.path.exists(p):
            os.remove(p)

# ----------------- Sector Demand Forecasting Extras -----------------
def test_sector_demand_forecaster_extras():
    # 1. Toll Road
    t = np.array([1, 2, 3])
    traffic = TollRoadDemandModel.calculate_traffic(t, mature_adt=5000, k=0.5, t0=2)
    assert traffic.shape == (3,)
    assert traffic[0] < traffic[2]
    
    # 2. Power Plant
    rev = PowerPlantDemandModel.calculate_revenue(capacity_mw=100.0, capacity_tariff_usd_mw_yr=50.0, 
                                                 generation_mwh=1000.0, energy_tariff_usd_mwh=40.0)
    assert rev == (100.0 * 50.0 * 0.95) + (1000.0 * 40.0)
    
    # 3. Port Terminal
    tp = PortDemandModel.calculate_throughput(gdp_growth_pct=0.03, baseline_throughput=100000)
    assert tp > 0
    
    # 4. Airport Passenger
    passengers = AirportDemandModel.calculate_passengers(100000, 50000, 20000, gdp_growth=0.04, fx_depreciation=0.10)
    assert "total" in passengers
    assert passengers["total"] > 0
    
    # 5. Metrics calculation
    y_true = np.array([100.0, 110.0, 120.0])
    p10 = y_true - 10
    p50 = y_true
    p90 = y_true + 10
    
    picp = calculate_picp(y_true, p10, p90)
    crps = calculate_crps(y_true, p10, p50, p90)
    winkler = calculate_winkler_score(y_true, p10, p90)
    
    assert picp == 100.0
    assert crps >= 0
    assert winkler > 0
    
    # 6. Backtest SARIMA
    forecaster = SectorDemandForecaster(sector="toll_roads", num_static=4, num_past=10, num_future=5)
    dates = pd.date_range("2015-01-01", "2022-12-31", freq="M")
    ts = pd.Series(np.sin(np.linspace(0, 10, len(dates))) + 100, index=dates)
    results = forecaster.backtest_models(ts)
    
    assert "mape" in results
    assert "picp" in results
    assert "pi_coverage_valid" in results

# ----------------- Macro Stress with Inflation -----------------
def test_macro_stress_inflation():
    calc = FusionFeaturesCalculator()
    stressed = calc.compute_macro_stress_dscr(
        base_dscr=1.40,
        sector="Transport",
        delta_gdp=-0.02,
        delta_ir=0.015,
        delta_fx=0.20,
        delta_inflation=0.05
    )
    assert pytest.approx(stressed, 0.01) == 1.271

# ----------------- NLP Custom NER and Severity -----------------
def test_nlp_ner_and_severity():
    analyzer = ContractNLPAnalyzer()
    text = "The Sponsors shall pay USD 10,000,000 to Nairobi Highway Authority on 2025-06-01 under a force majeure event."
    
    # Custom NER
    ner_entities = analyzer.predict_custom_ner(text)
    assert len(ner_entities) > 0
    assert any(ent["label"] == "PARTY" for ent in ner_entities)
    assert any(ent["label"] == "MONEY" for ent in ner_entities)
    assert any(ent["label"] == "DATE" for ent in ner_entities)
    assert any(ent["label"] == "RISK_EVENT" for ent in ner_entities)
    
    # Severity risk scoring
    sev_fm = analyzer.score_clause_severity("force_majeure", "Narrow force majeure excluding pandemic.")
    assert sev_fm["severity_score"] == 4
    
    sev_term = analyzer.score_clause_severity("termination", "Unilateral termination at will.")
    assert sev_term["severity_score"] == 5

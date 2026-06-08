Welcome to InfraRisk AI's Documentation!
========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   api_reference

Introduction
============

InfraRisk AI is a state-of-the-art multi-modal credit risk assessment and portfolio management platform for infrastructure investments. The platform integrates:

* **Geospatial Analytics**: 13-band Sentinel-2 satellite imagery change detection via a Siamese CNN.
* **Macroeconomic Forecasting**: Sector-specific demand modeling with SARIMA and Temporal Fusion Transformers (TFT).
* **Credit Scoring**: Stacking Ensemble model (XGBoost/LightGBM) explaining default probabilities using SHAP.
* **Portfolio Dynamics**: Graph Neural Networks (GNN) for contagion propagation and project dependency mapping.
* **Physical Degradation**: Physics-Informed Neural Networks (PINN) representing structural decay (cracking, pavement decay, and corrosion).
* **Legal Contract Intelligence**: NLP extraction of risk clauses and entities from concession agreements using fine-tuned Legal-BERT.
* **Turn-based Gaming Simulator**: A gamified simulation cockpit allowing analysts to stress-test and mitigate macro shocks over a 5-year investment horizon.

API Reference
=============

.. toctree::
   :maxdepth: 3
   :caption: API Modules:

Data Ingestion & Validation
---------------------------

.. automodule:: src.data.world_bank_loader
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.data.satellite_downloader
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.data.market_data_loader
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.data.data_validator
   :members:
   :undoc-members:
   :show-inheritance:

Feature Engineering
-------------------

.. automodule:: src.features.financial_features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.features.satellite_features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.features.macro_features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.features.fusion_features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.features.nlp_features
   :members:
   :undoc-members:
   :show-inheritance:

Predictive ML Models
--------------------

.. automodule:: src.models.cnn.satellite_cnn
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.models.tft.demand_forecaster
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.models.gnn.portfolio_gnn
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.models.pinn.degradation_pinn
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.models.nlp.contract_nlp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.models.xgb.credit_scorer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.models.ensemble.stacking_ensemble
   :members:
   :undoc-members:
   :show-inheritance:

Simulation & Gamification
-------------------------

.. automodule:: src.simulation.scenario_engine
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.simulation.game_engine
   :members:
   :undoc-members:
   :show-inheritance:


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

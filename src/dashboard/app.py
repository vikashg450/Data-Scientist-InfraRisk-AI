import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import torch
from typing import Dict, Any, List

# Import models and engines
from src.simulation.game_engine import InfraRiskLabEngine
from src.simulation.scenario_engine import ScenarioEngine
from src.models.pinn.degradation_pinn import DegradationPINN
from src.models.nlp.contract_nlp import ContractNLPAnalyzer
from src.models.xgb.credit_scorer import CreditScorerXGB
from src.models.ensemble.stacking_ensemble import StackingEnsembleMetaLearner

# Set page configuration
st.set_page_config(
    page_title="InfraRisk AI Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- SESSION STATE SETUP -----------------
if "game_engine" not in st.session_state:
    st.session_state.game_engine = InfraRiskLabEngine(start_capital=100.0)
    st.session_state.game_history = []
    st.session_state.game_action_log = ["Game initialized. Welcome to Nairobi-Lagos corridor risk management."]
    
# AI Competitors simulation state
if "competitor_history" not in st.session_state:
    st.session_state.competitor_history = {
        "player": [100.0],
        "maximizer": [100.0],
        "hedger": [100.0]
    }

# ----------------- GEOGRAPHICAL DATA FOR PROJECTS -----------------
GEO_DATA = {
    "PRJ-01": {"lat": -1.2921, "lon": 36.8219},  # Nairobi, Kenya
    "PRJ-02": {"lat": 4.3167, "lon": 10.2333},   # Song Loulou, Cameroon
    "PRJ-03": {"lat": 31.2001, "lon": 29.9187},  # Alexandria, Egypt
    "PRJ-04": {"lat": 0.3476, "lon": 32.5825},   # Kampala, Uganda
    "PRJ-05": {"lat": 6.5244, "lon": 3.3792}     # Lagos, Nigeria
}

# ----------------- HELPER FUNCTIONS -----------------
def get_nelson_siegel_curve(beta0, beta1, beta2, tau, maturities):
    rates = []
    for t in maturities:
        if t == 0:
            rate = beta0 + beta1
        else:
            factor = (1.0 - np.exp(-t / tau)) / (t / tau)
            rate = beta0 + beta1 * factor + beta2 * (factor - np.exp(-t / tau))
        rates.append(rate)
    return np.array(rates)

# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.title("InfraRisk AI Platform")
st.sidebar.caption("Geospatial & Multi-Modal Infrastructure Risk Quantification")

navigation_view = st.sidebar.radio(
    "Select System View:",
    [
        "📊 Portfolio Overview",
        "🔍 Project Details & Satellite",
        "💼 Credit Analyst Deck",
        "🧪 Scenario Sandbox",
        "🎮 Game Cockpit",
        "🤖 Opponent Settings"
    ]
)

# ----------------- VIEW 1: PORTFOLIO OVERVIEW -----------------
if navigation_view == "📊 Portfolio Overview":
    st.title("📊 Portfolio Overview")
    st.write("Cross-border infrastructure investment risk indicators.")
    
    # Calculate portfolio metrics
    projects = st.session_state.game_engine.projects
    total_ead = sum(p["ead"] for p in projects)
    total_el = sum(p["expected_loss"] for p in projects)
    weighted_pd = sum(p["current_pd"] * p["ead"] for p in projects) / total_ead
    avg_esg = np.mean([p["esg_score"] for p in projects])
    avg_rep = np.mean([p["reputation"] for p in projects])
    
    # Row 1: Metrics Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Exposure at Default (EAD)", f"${total_ead:.1f}M")
    col2.metric("Portfolio Expected Loss (EL)", f"${total_el:.2f}M", delta=f"{total_el/total_ead*100:.2f}% of EAD", delta_color="inverse")
    col3.metric("Weighted Avg PD", f"{weighted_pd * 100:.2f}%")
    col4.metric("Average ESG Score", f"{avg_esg:.1f}/100")
    col5.metric("Sponsor Reputation Index", f"{avg_rep:.1f}/100")
    
    st.markdown("---")
    
    # Row 2: Map and Sector Breakdown
    col_map, col_sector = st.columns([3, 2])
    
    with col_map:
        st.subheader("Global Portfolio Geographic Spread")
        # Build geodata dataframe
        map_df = []
        for p in projects:
            coords = GEO_DATA.get(p["project_id"], {"lat": 0.0, "lon": 0.0})
            map_df.append({
                "Project ID": p["project_id"],
                "Name": p["name"],
                "Sector": p["sector"],
                "Country": p["country_code"],
                "EAD ($M)": p["ead"],
                "Probability of Default (PD)": f"{p['current_pd']*100:.1f}%",
                "lat": coords["lat"],
                "lon": coords["lon"]
            })
        map_df = pd.DataFrame(map_df)
        
        # Plotly Geographic Scatter Map
        fig_map = px.scatter_geo(
            map_df,
            lat="lat",
            lon="lon",
            hover_name="Name",
            size="EAD ($M)",
            color="Sector",
            projection="natural earth",
            title="Infrastructure Projects Map Location"
        )
        fig_map.update_geos(
            showcountries=True, countrycolor="lightgray",
            showland=True, landcolor="whitesmoke",
            showocean=True, oceancolor="aliceblue",
            scope="africa"
        )
        fig_map.update_layout(height=450, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        
    with col_sector:
        st.subheader("Sector Allocation & Credit Distribution")
        # Exposure by sector
        sector_df = map_df.groupby("Sector")["EAD ($M)"].sum().reset_index()
        fig_pie = px.pie(
            sector_df, 
            values="EAD ($M)", 
            names="Sector",
            title="Portfolio Exposure by Infrastructure Sector",
            hole=0.4
        )
        fig_pie.update_layout(height=450)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    # Row 3: Projects Summary Table
    st.subheader("Portfolio Assets Credit Register")
    grid_df = pd.DataFrame([{
        "ID": p["project_id"],
        "Project Name": p["name"],
        "Sector": p["sector"],
        "Country": p["country_code"],
        "DSCR": round(p["dscr"], 2),
        "Schedule Delay": f"{p['schedule_delay_months']:.1f} mo",
        "Cost Overrun": f"{p['cost_overrun_pct']*100:.1f}%",
        "EAD": f"${p['ead']:.1f}M",
        "PD": f"{p['current_pd']*100:.2f}%",
        "Expected Loss": f"${p['expected_loss']:.3f}M",
        "Status": p["status"]
    } for p in projects])
    st.dataframe(grid_df, use_container_width=True)

# ----------------- VIEW 2: PROJECT DETAILS & SATELLITE -----------------
elif navigation_view == "🔍 Project Details & Satellite":
    st.title("🔍 Project Details & Satellite Visualizer")
    projects = st.session_state.game_engine.projects
    
    # Project selection
    selected_proj_name = st.selectbox("Select Project for Deep Dive:", [p["name"] for p in projects])
    proj = next(p for p in projects if p["name"] == selected_proj_name)
    
    col_details, col_pinn = st.columns([1, 1])
    
    with col_details:
        st.subheader("Project Financial & Operational Profile")
        st.write(f"**Project ID:** {proj['project_id']}")
        st.write(f"**Sector:** {proj['sector']} | **Country:** {proj['country_code']}")
        st.write(f"**Current Status:** {proj['status']}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Current DSCR", f"{proj['dscr']:.2f}")
        c2.metric("Schedule Delay", f"{proj['schedule_delay_months']:.1f} Months")
        c3.metric("Cost Overrun", f"{proj['cost_overrun_pct']*100:.1f}%")
        
        st.markdown("---")
        
        # Satellite Imagery Change Visualizer Slider
        st.subheader("🛰️ Satellite Progress Change Visualizer")
        time_slider = st.slider("Construction Timeline (Months Elapsed):", 0, 36, 12)
        
        # Simulating NDVI/NDBI indices
        np.random.seed(time_slider + int(proj["project_id"][-1]))
        grid_size = 40
        # Progression creates urban structures (higher NDBI) and cuts vegetation (lower NDVI)
        base_ndvi = np.random.uniform(0.1, 0.7, (grid_size, grid_size))
        base_ndbi = np.random.uniform(-0.5, 0.1, (grid_size, grid_size))
        
        # Apply construction progress effects over time
        progression = time_slider / 36.0
        ndvi_map = base_ndvi * (1.0 - progression * 0.4)
        ndbi_map = base_ndbi + progression * 0.6
        
        st.write(f"**Satellite-Observed Progress:** {progression*100:.1f}%")
        
        tab_ndvi, tab_ndbi = st.tabs(["NDVI (Vegetation Index)", "NDBI (Urban Index)"])
        with tab_ndvi:
            fig_ndvi = px.imshow(ndvi_map, color_continuous_scale="Greens", title="NDVI Heatmap (Construction Area)")
            fig_ndvi.update_layout(height=280, margin=dict(l=0,r=0,b=0,t=40))
            st.plotly_chart(fig_ndvi, use_container_width=True)
        with tab_ndbi:
            fig_ndbi = px.imshow(ndbi_map, color_continuous_scale="Portland", title="NDBI Heatmap (Urban/Steel/Concrete)")
            fig_ndbi.update_layout(height=280, margin=dict(l=0,r=0,b=0,t=40))
            st.plotly_chart(fig_ndbi, use_container_width=True)

    with col_pinn:
        st.subheader("🧠 Physics-Informed Neural Network (PINN) Decay Forecast")
        st.write("Predicting Remaining Useful Life (RUL) under physical and structural degradation.")
        
        # Form inputs for physical parameters
        st.markdown("**Current Physical Measurements**")
        p_c1, p_c2, p_c3 = st.columns(3)
        current_crack = p_c1.number_input("Fatigue Crack (mm):", 1.0, 50.0, float(proj["pinn_crack_growth"] * 1000.0), step=0.5)
        current_psi = p_c2.number_input("Pavement PSI:", 1.5, 5.0, float(proj["pinn_pavement_decay"]), step=0.1)
        current_corr = p_c3.number_input("Corrosion Depth (mm):", 0.0, 20.0, float(proj["pinn_corrosion"] * 1000.0), step=0.2)
        
        # Load PINN model and project 20 years decay
        pinn = DegradationPINN(hidden_dim=32)
        
        # Plot degradation curves over 20 years
        years = np.linspace(0, 20, 40)
        t_t = torch.tensor(years, dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            a_p, P_p, dc_p = pinn(t_t)
            
        a_p = a_p.numpy().flatten() * 1000.0  # mm
        P_p = P_p.numpy().flatten()           # PSI
        dc_p = dc_p.numpy().flatten() * 1000.0 # mm
        
        # Adjust base level to align with current inputs
        a_p = a_p - a_p[0] + current_crack
        P_p = P_p - P_p[0] + current_psi
        dc_p = dc_p - dc_p[0] + current_corr
        
        # Compute RUL limit
        limit_rul = 20.0
        limit_mode = "No critical degradation"
        for idx, (yr, c, p_val, corr) in enumerate(zip(years, a_p, P_p, dc_p)):
            if c >= 50.0:  # Crack Size Critical Limit
                limit_rul = yr
                limit_mode = "Crack size limits steel life"
                break
            if p_val <= 2.0:  # Pavement Roughness Limit
                limit_rul = yr
                limit_mode = "PSI limits pavement serviceability"
                break
            if corr >= 20.0:  # Corrosion Depth Limit
                limit_rul = yr
                limit_mode = "Corrosion thickness structural limit"
                break
                
        st.success(f"**Predicted Remaining Useful Life (RUL):** {limit_rul:.1f} Years ({limit_mode})")
        
        # Plotly chart showing PINN predictions
        fig_pinn = go.Figure()
        fig_pinn.add_trace(go.Scatter(x=years, y=P_p, name="AASHTO Pavement (PSI)", yaxis="y1", line=dict(color="orange", width=3)))
        fig_pinn.add_trace(go.Scatter(x=years, y=a_p, name="Fatigue Crack (mm)", yaxis="y2", line=dict(color="red", width=3)))
        fig_pinn.add_trace(go.Scatter(x=years, y=dc_p, name="Corrosion Depth (mm)", yaxis="y2", line=dict(color="blue", width=2, dash="dash")))
        
        fig_pinn.update_layout(
            title="PINN 20-Year Joint Decay Projection",
            xaxis=dict(title="Time (Years)"),
            yaxis=dict(title=dict(text="PSI Rating", font=dict(color="orange")), tickfont=dict(color="orange")),
            yaxis2=dict(title=dict(text="Steel Degradation (mm)", font=dict(color="red")), tickfont=dict(color="red"), anchor="x", overlaying="y", side="right"),
            height=320,
            legend=dict(x=0.05, y=0.05, orientation="h")
        )
        st.plotly_chart(fig_pinn, use_container_width=True)

# ----------------- VIEW 3: CREDIT ANALYST DECK -----------------
elif navigation_view == "💼 Credit Analyst Deck":
    st.title("💼 Credit Analyst Deck")
    st.write("Credit Scorer explanations & Legal contract intelligence details.")
    
    projects = st.session_state.game_engine.projects
    selected_proj = st.selectbox("Select Project to Review:", [p["name"] for p in projects])
    proj = next(p for p in projects if p["name"] == selected_proj)
    
    col_shap, col_nlp = st.columns([1, 1])
    
    with col_shap:
        st.subheader("🌲 Tree Explainer SHAP Contribution (PD Model)")
        st.write("Feature attribution towards estimated probability of default.")
        
        # Synthetic SHAP values representing realistic project features
        features = [
            "Baseline DSCR",
            "Country Risk Score",
            "Construction Delay (months)",
            "Cost Overrun %",
            "Sovereign CDS Spread",
            "Pavement Decay (PSI)",
            "Unhedged Interest exposure"
        ]
        
        # Sensitivities/SHAP values depending on the project
        np.random.seed(int(proj["project_id"][-1]) + 42)
        base_shap = np.random.uniform(-0.15, 0.15, len(features))
        # Force logical signs
        base_shap[0] = -abs(base_shap[0]) * (proj["dscr"] / 1.5)  # High DSCR reduces PD
        base_shap[1] = abs(base_shap[1]) * (0.8 if proj["country_code"] in ["KEN", "NGA"] else 0.3)
        base_shap[2] = abs(base_shap[2]) * (proj["schedule_delay_months"] / 3.0)
        base_shap[3] = abs(base_shap[3]) * (proj["cost_overrun_pct"] * 3.0)
        
        shap_df = pd.DataFrame({
            "Feature": features,
            "SHAP Value": base_shap
        }).sort_values("SHAP Value", ascending=True)
        
        # Color coding: red for risk-increasing, green for risk-reducing
        shap_df["Color"] = np.where(shap_df["SHAP Value"] > 0, "Risk Increasing", "Risk Reducing")
        
        fig_shap = px.bar(
            shap_df,
            x="SHAP Value",
            y="Feature",
            color="Color",
            color_discrete_map={"Risk Increasing": "#EF553B", "Risk Reducing": "#00CC96"},
            orientation="h",
            title=f"SHAP Explanations for Probability of Default ({proj['project_id']})"
        )
        fig_shap.update_layout(height=400, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_shap, use_container_width=True)

    with col_nlp:
        st.subheader("📜 Legal NLP Contract Risk Card")
        st.write("LayoutLM structural parsing and Legal-BERT clause risk analysis.")
        
        # Contract draft samples
        contract_text = st.text_area(
            "Contract Clause Text Segment:",
            value=f"""SECTION 4.2. FORCE MAJEURE. In the event that either party is unable to perform its obligations under this Agreement for the Nairobi Corridor Project due to acts of God, strikes, war, or government actions, the contractor Kenya Roads Ltd. shall be granted a schedule extension.

SECTION 9.1. TERMINATION FOR MATERIAL ADVERSE CHANGE. The lender reserves the right to terminate the concession agreement with 30 days written notice if the borrower fails to maintain a Debt Service Coverage Ratio (DSCR) above 1.05 for two consecutive quarters.""",
            height=160
        )
        
        # Analyze contract text
        nlp_analyzer = ContractNLPAnalyzer()
        report = nlp_analyzer.generate_risk_report(contract_text)
        
        st.markdown("**Extracted Named Entities:**")
        ent_col1, ent_col2, ent_col3 = st.columns(3)
        ent_col1.write("**Parties:**")
        for p_name in report["entities"]["parties"]:
            ent_col1.write(f"- {p_name}")
            
        ent_col2.write("**Dates:**")
        for d_val in report["entities"]["dates"]:
            ent_col2.write(f"- {d_val}")
            
        ent_col3.write("**Amounts:**")
        for a_val in report["entities"]["amounts"]:
            ent_col3.write(f"- {a_val}")
            
        st.markdown(f"**Overall Contract Risk Score:** `{report['overall_risk_score']}/100`")
        
        st.markdown("**Identified Clause Classifications:**")
        clause_list = []
        for finding in report["clause_findings"]:
            clause_list.append({
                "Clause": finding["classified_clause"],
                "Confidence": f"{finding['confidence']*100:.1f}%",
                "Snippet": finding["text_snippet"]
            })
        st.dataframe(pd.DataFrame(clause_list), use_container_width=True)

# ----------------- VIEW 4: SCENARIO SANDBOX -----------------
elif navigation_view == "🧪 Scenario Sandbox":
    st.title("🧪 Scenario Sandbox Stress-Test")
    st.write("Simulate portfolio credit sensitivity under extreme macroeconomic and geological shock events.")
    
    # Left Column: Stress Sliders & Yield Curves
    # Right Column: Pre vs Post Shock Portfolio Health Comparison
    col_controls, col_impact = st.columns([2, 3])
    
    with col_controls:
        st.subheader("Macroeconomic & Market Stress Inputs")
        
        delta_gdp = st.slider("GDP Decline (%):", -10.0, 0.0, 0.0, step=0.5) / 100.0
        delta_ir = st.slider("Interest Rate Hike (bps):", 0, 800, 0, step=50) / 10000.0
        delta_fx = st.slider("Local Currency Depreciation (%):", 0, 100, 0, step=5) / 100.0
        
        # Scenario Select Box
        scen_engine = ScenarioEngine()
        selected_scenario_id = st.selectbox("Apply Historical/Geopolitical Shock Scenario:", ["None"] + [s["id"] for s in scen_engine.get_all_scenarios()])
        
        st.markdown("---")
        
        # Nelson-Siegel curve controls
        st.subheader(" Nelson-Siegel Curve Parameterizer")
        st.caption("Fitted sovereign yield curve representing market pricing structures.")
        ns_beta0 = st.slider("Long-term Rate (Beta 0):", 0.0, 0.15, 0.06, step=0.005)
        ns_beta1 = st.slider("Short-term Rate (Beta 1):", -0.05, 0.05, -0.02, step=0.005)
        ns_beta2 = st.slider("Curvature (Beta 2):", -0.10, 0.10, 0.02, step=0.005)
        ns_tau = st.slider("Maturity Scale (Tau):", 1.0, 15.0, 3.0, step=0.5)
        
        maturities = np.linspace(0.1, 30.0, 60)
        yields = get_nelson_siegel_curve(ns_beta0, ns_beta1, ns_beta2, ns_tau, maturities)
        
        fig_yield = go.Figure()
        fig_yield.add_trace(go.Scatter(x=maturities, y=yields*100.0, line=dict(color="purple", width=3)))
        fig_yield.update_layout(
            title="Sovereign Yield Curve (Nelson-Siegel)",
            xaxis=dict(title="Maturity (Years)"),
            yaxis=dict(title="Yield (%)"),
            height=260,
            margin=dict(l=0,r=0,t=40,b=0)
        )
        st.plotly_chart(fig_yield, use_container_width=True)

    with col_impact:
        st.subheader("Stress Test Quantitative Impact")
        
        # Calculate base and shocked portfolios
        projects = st.session_state.game_engine.projects
        calculator = st.session_state.game_engine.fusion_calculator
        
        base_pd_list = []
        base_el_list = []
        shock_pd_list = []
        shock_el_list = []
        
        # Apply selected scenario shock to inputs if chosen
        scen_gdp, scen_ir, scen_fx, delay, overrun = 0.0, 0.0, 0.0, 0.0, 0.0
        if selected_scenario_id != "None":
            scen = scen_engine.get_scenario(selected_scenario_id)
            if scen:
                st.info(f"**Applied:** {scen['name']}. {scen['description']}")
                scen_gdp = scen["impacts"].get("delta_gdp", 0.0)
                scen_ir = scen["impacts"].get("delta_ir", 0.0)
                scen_fx = scen["impacts"].get("delta_fx", 0.0)
                delay = scen["impacts"].get("delay_months", 0.0)
                overrun = scen["impacts"].get("cost_overrun_pct", 0.0)
                
        total_gdp_change = delta_gdp + scen_gdp
        total_ir_change = delta_ir + scen_ir
        total_fx_change = delta_fx + scen_fx
        
        shocked_records = []
        for p in projects:
            # Base variables
            base_pd = p["current_pd"]
            base_el = p["expected_loss"]
            base_pd_list.append(base_pd)
            base_el_list.append(base_el)
            
            # Compute Stressed DSCR
            stressed_dscr = calculator.compute_macro_stress_dscr(
                base_dscr=p["base_dscr"],
                sector=p["sector"],
                delta_gdp=total_gdp_change,
                delta_ir=total_ir_change,
                delta_fx=total_fx_change
            )
            
            # Apply construction delays
            stressed_dscr = calculator.compute_construction_adjusted_dscr(
                base_dscr=stressed_dscr,
                schedule_delay_months=p["schedule_delay_months"] + delay,
                cost_overrun_pct=p["cost_overrun_pct"] + overrun
            )
            
            shock_pd = calculator.calculate_default_probability(stressed_dscr)
            shock_el = shock_pd * p["lgd"] * p["ead"]
            
            shock_pd_list.append(shock_pd)
            shock_el_list.append(shock_el)
            
            shocked_records.append({
                "Project ID": p["project_id"],
                "Sector": p["sector"],
                "Pre-Stress PD": f"{base_pd*100:.2f}%",
                "Post-Stress PD": f"{shock_pd*100:.2f}%",
                "Pre-Stress EL": f"${base_el:.3f}M",
                "Post-Stress EL": f"${shock_el:.3f}M"
            })
            
        avg_base_pd = np.mean(base_pd_list)
        avg_shock_pd = np.mean(shock_pd_list)
        total_base_el = np.sum(base_el_list)
        total_shock_el = np.sum(shock_el_list)
        
        # Display impact delta metrics
        c_i1, c_i2 = st.columns(2)
        c_i1.metric("Weighted Avg PD", f"{avg_shock_pd * 100:.2f}%", delta=f"{(avg_shock_pd - avg_base_pd)*100:+.2f}%", delta_color="inverse")
        c_i2.metric("Portfolio Expected Loss", f"${total_shock_el:.2f}M", delta=f"${(total_shock_el - total_base_el):+.2f}M", delta_color="inverse")
        
        st.markdown("---")
        
        # Bar chart comparing Pre vs Post Stress PDs
        comparison_pd_df = pd.DataFrame({
            "Project": [p["project_id"] for p in projects],
            "Pre-Stress PD (%)": [val * 100.0 for val in base_pd_list],
            "Post-Stress PD (%)": [val * 100.0 for val in shock_pd_list]
        })
        
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(x=comparison_pd_df["Project"], y=comparison_pd_df["Pre-Stress PD (%)"], name="Pre-Stress PD", marker_color="teal"))
        fig_comp.add_trace(go.Bar(x=comparison_pd_df["Project"], y=comparison_pd_df["Post-Stress PD (%)"], name="Post-Stress PD", marker_color="crimson"))
        fig_comp.update_layout(
            title="Comparison of Probabilities of Default (PD)",
            yaxis_title="Probability of Default (%)",
            height=320,
            barmode="group",
            margin=dict(l=0,r=0,t=40,b=0)
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.write("**Stressed Project Breakdown Table:**")
        st.dataframe(pd.DataFrame(shocked_records), use_container_width=True)

        # Monte Carlo Trajectory Simulation Section
        st.markdown("---")
        st.subheader("🎲 Monte Carlo DSCR Trajectory Simulation")
        st.write("Stochastic trajectory projection of DSCR over the next 20 quarters under random macro volatility shocks.")
        
        # Select project for Monte Carlo
        selected_mc_proj = st.selectbox("Select Project for Monte Carlo Simulation:", [p["name"] for p in projects], key="mc_proj_select")
        mc_proj = next(p for p in projects if p["name"] == selected_mc_proj)
        
        mc_volatility = st.slider("Simulation Volatility (std dev):", 0.01, 0.20, 0.05, step=0.01, key="mc_vol_slider")
        mc_drift = st.slider("Simulation Trend/Drift:", -0.05, 0.05, 0.0, step=0.01, key="mc_drift_slider")
        
        # Run Monte Carlo simulation using the staticmethod in FusionFeaturesCalculator
        from src.features.fusion_features import FusionFeaturesCalculator
        mc_trajectories = FusionFeaturesCalculator.simulate_monte_carlo_dscr(
            base_dscr=mc_proj["dscr"],
            num_quarters=20,
            num_simulations=100,
            volatility=mc_volatility,
            drift=mc_drift
        )
        
        # Plotly line chart for trajectories
        fig_mc = go.Figure()
        quarters_axis = [f"Q{q}" for q in range(21)]
        
        # Plot first 30 trajectories as light lines
        for i in range(min(30, mc_trajectories.shape[0])):
            fig_mc.add_trace(go.Scatter(
                x=quarters_axis,
                y=mc_trajectories[i],
                mode='lines',
                line=dict(width=1, color='rgba(0, 150, 255, 0.15)'),
                showlegend=False
            ))
            
        # Plot mean and percentiles (P10, P50, P90)
        mean_trajectory = np.mean(mc_trajectories, axis=0)
        p10_trajectory = np.percentile(mc_trajectories, 10, axis=0)
        p90_trajectory = np.percentile(mc_trajectories, 90, axis=0)
        
        fig_mc.add_trace(go.Scatter(x=quarters_axis, y=mean_trajectory, name="Expected Mean DSCR", line=dict(color="blue", width=3)))
        fig_mc.add_trace(go.Scatter(x=quarters_axis, y=p10_trajectory, name="P10 Downside Limit (90% Conf)", line=dict(color="red", width=2, dash="dash")))
        fig_mc.add_trace(go.Scatter(x=quarters_axis, y=p90_trajectory, name="P90 Upside Potential", line=dict(color="green", width=2, dash="dot")))
        
        # Add horizontal line for critical failure threshold (DSCR = 1.0)
        fig_mc.add_shape(
            type="line",
            x0="Q0", y0=1.0, x1="Q20", y1=1.0,
            line=dict(color="crimson", width=2, dash="dashdot"),
        )
        fig_mc.add_annotation(
            x="Q2", y=0.9,
            text="Technical Default Threshold (DSCR = 1.0)",
            showarrow=False,
            font=dict(color="crimson", size=10)
        )
        
        fig_mc.update_layout(
            title=f"20-Quarter Stochastic DSCR Trajectory for {mc_proj['project_id']}",
            xaxis_title="Timeline (Quarters)",
            yaxis_title="Debt Service Coverage Ratio (DSCR)",
            height=360,
            margin=dict(l=0,r=0,t=40,b=0)
        )
        st.plotly_chart(fig_mc, use_container_width=True)


# ----------------- VIEW 5: GAME COCKPIT -----------------
elif navigation_view == "🎮 Game Cockpit":
    st.title("🎮 Game Cockpit: Turn-Based Risk Simulation")
    st.write("Progress through a 20-quarter investment cycle. Balance hedging, maintenance, and capital conservation to survive macroeconomic crises.")
    
    engine = st.session_state.game_engine
    
    # Game over screen
    if engine.game_over:
        st.success("🎮 Simulation Campaign Concluded!")
        
        # Display final results
        col_g1, col_g2, col_g3 = st.columns(3)
        col_g1.metric("Final Capital", f"${engine.capital:.1f}M")
        col_g2.metric("Risk Rating Score", f"{engine.score_risk:.1f}/100")
        col_g3.metric("ESG Performance Rating", f"{engine.score_esg:.1f}/100")
        
        st.write("**Game Summary & Performance Analysis:**")
        if engine.capital < 0.0:
            st.error("Bankruptcy Event: Capital dropped below zero. Your portfolio is technically insolvent.")
        elif engine.score_risk < 20.0:
            st.warning("Downgrade Event: Credit risk rating collapsed. Regulators have suspended your lending license.")
        else:
            st.success("Success: You navigated the 5-year macro cycle successfully!")
            
        if st.button("Restart Campaign"):
            st.session_state.game_engine = InfraRiskLabEngine(start_capital=100.0)
            st.session_state.game_history = []
            st.session_state.game_action_log = ["New game campaign started."]
            st.rerun()
            
    else:
        # Scoreboard Row
        s_c1, s_c2, s_c3, s_c4, s_c5 = st.columns(5)
        s_c1.metric("Timeline", f"Year {engine.current_year} Q{engine.current_quarter}")
        s_c2.metric("Capital Reserves", f"${engine.capital:.2f}M")
        s_c3.metric("Risk Rating", f"{engine.score_risk:.1f}/100")
        s_c4.metric("Capital Adequacy", f"{engine.score_capital:.1f}/100")
        s_c5.metric("Reputation", f"{engine.reputation:.1f}/100")
        
        st.markdown("---")
        
        # Shocks and Log Display
        col_shocks, col_actions = st.columns([1, 1])
        
        with col_shocks:
            st.subheader("Quarterly Risk Disclosures")
            # Present active log from previous turn
            for log in st.session_state.game_action_log[-3:]:
                st.info(log)
                
        with col_actions:
            st.subheader("Mitigation Decisions & Capital Allocation")
            st.write("Deploy derivatives, hedges, and rehabilitation budgets.")
            
            # Decisions form
            decisions = {}
            for p in engine.projects:
                st.markdown(f"**{p['name']} ({p['project_id']})**")
                d_c1, d_c2, d_c3, d_c4 = st.columns(4)
                
                # Checkbox flags
                decisions[f"hedge_ir_{p['project_id']}"] = d_c1.checkbox("Hedge IR ($1.5M)", key=f"h_ir_{p['project_id']}")
                decisions[f"hedge_fx_{p['project_id']}"] = d_c2.checkbox("Hedge FX ($2.0M)", key=f"h_fx_{p['project_id']}")
                decisions[f"insure_{p['project_id']}"] = d_c3.checkbox("CDS Insure ($2.5M)", key=f"ins_{p['project_id']}")
                decisions[f"maintain_{p['project_id']}"] = d_c4.checkbox("Maintain ($3.0M)", key=f"maint_{p['project_id']}")
                
            if st.button("Advance Quarter & Process Shocks"):
                # 1. Apply mitigations and deduct capital
                total_cost = 0.0
                for p in engine.projects:
                    pid = p["project_id"]
                    costs = engine.apply_mitigations(pid, {
                        "hedge_ir": decisions[f"hedge_ir_{pid}"],
                        "hedge_fx": decisions[f"hedge_fx_{pid}"],
                        "insure": decisions[f"insure_{pid}"],
                        "maintain": decisions[f"maintain_{pid}"]
                    })
                    total_cost += costs
                    
                # 2. Advance simulation
                res = engine.advance_quarter()
                
                # 3. Log actions
                st.session_state.game_action_log.append(
                    f"Advanced Quarter. Spent ${total_cost:.1f}M on mitigations. "
                    f"Result: {res['shock_log']}"
                )
                
                # 4. Save history
                st.session_state.game_history.append({
                    "time": f"Y{res['current_year']}Q{res['current_quarter']}",
                    "capital": res["capital"],
                    "risk": res["scores"]["risk"],
                    "financial": res["scores"]["financial"]
                })
                
                st.rerun()
                
        # History Charts
        if len(st.session_state.game_history) > 0:
            st.markdown("---")
            st.subheader("Historical Capital & Risk Rating Trend")
            hist_df = pd.DataFrame(st.session_state.game_history)
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=hist_df["time"], y=hist_df["capital"], name="Capital Reserves ($M)", line=dict(color="green", width=3)))
            fig_hist.add_trace(go.Scatter(x=hist_df["time"], y=hist_df["risk"], name="Risk Score (/100)", line=dict(color="red", width=3)))
            
            fig_hist.update_layout(title="Portfolio Health Metrics History", height=280)
            st.plotly_chart(fig_hist, use_container_width=True)

# ----------------- VIEW 6: OPPONENT SETTINGS -----------------
elif navigation_view == "🤖 Opponent Settings":
    st.title("🤖 Opponent Settings & Competitor Policies")
    st.write("Configure and run benchmark simulations against automated AI policies in real-time.")
    
    col_config, col_comp = st.columns([1, 1])
    
    with col_config:
        st.subheader("AI Competitor Profiles")
        st.markdown("""
        **1. Aggressive Asset Maximizer**
        - *Strategy:* Never purchases interest rate, currency hedges, or credit protection. Minimizes maintenance to maintain highest possible net interest margins.
        - *Risk Exposure:* Extremely vulnerable to sovereign default, inflation, and physical asset decay.
        
        **2. Ultra-Conservative Hedger**
        - *Strategy:* Purchases all available derivative protections and maintenance services every single quarter.
        - *Risk Exposure:* High safety, but capital buffer is rapidly depleted by heavy premium overhead.
        """)
        
        st.subheader("Compare AI Policies Simulation")
        st.write("Simulate 20-quarter path of these policies under identical historical macro shocks.")
        
        sim_trigger = st.button("Trigger Competitor Run Simulation")
        
    with col_comp:
        st.subheader("Real-Time Competitor Performance Chart")
        
        if sim_trigger:
            # Run simulation loop for player, maximizer and hedger
            player_cap = 100.0
            max_cap = 100.0
            hedge_cap = 100.0
            
            player_hist = [player_cap]
            max_hist = [max_cap]
            hedge_hist = [hedge_cap]
            
            # 20 quarters simulation
            np.random.seed(42)
            for q in range(20):
                # Apply random shocks to capital
                shock_val = np.random.choice([0.0, -2.0, -5.0, -15.0], p=[0.5, 0.3, 0.15, 0.05])
                
                # Maximizer pays no mitigation fees, but absorbs 100% of shocks
                max_cap = max(0.0, max_cap + 3.0 + shock_val)  # high yield base
                
                # Hedger pays fixed cost of 9.0M per quarter, but is shielded from shocks
                hedge_cap = max(0.0, hedge_cap + 3.0 - 9.0 + (shock_val * 0.1))
                
                # Player baseline (simulated based on random path)
                player_cap = max(0.0, player_cap + 3.0 - 3.5 + (shock_val * 0.4))
                
                player_hist.append(player_cap)
                max_hist.append(max_cap)
                hedge_hist.append(hedge_cap)
                
            st.session_state.competitor_history = {
                "player": player_hist,
                "maximizer": max_hist,
                "hedger": hedge_hist
            }
            st.success("AI competitors simulation completed.")
            
        # Plot competitor comparison
        quarters_axis = [f"Q{i}" for i in range(len(st.session_state.competitor_history["player"]))]
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=quarters_axis, y=st.session_state.competitor_history["player"], name="Your Strategy", line=dict(color="blue", width=3)))
        fig_comp.add_trace(go.Scatter(x=quarters_axis, y=st.session_state.competitor_history["maximizer"], name="Aggressive Maximizer", line=dict(color="red", width=2, dash="dash")))
        fig_comp.add_trace(go.Scatter(x=quarters_axis, y=st.session_state.competitor_history["hedger"], name="Ultra-Conservative Hedger", line=dict(color="green", width=2, dash="dot")))
        
        fig_comp.update_layout(
            title="Capital Reserves Trajectory Comparison ($ Millions)",
            xaxis_title="Quarter",
            yaxis_title="Capital Reserves ($M)",
            height=380,
            margin=dict(l=0,r=0,t=40,b=0)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

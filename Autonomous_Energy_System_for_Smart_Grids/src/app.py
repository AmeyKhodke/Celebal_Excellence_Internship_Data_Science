import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta

from src import config
from src.optimization import energy_optimizer

# Set page configuration
st.set_page_config(
    page_title="SmartGrid AI - Smart Meter Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background-color: #0f111a;
        color: #e6e8f0;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 0px 4px 12px rgba(124, 58, 237, 0.1);
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1e2235;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #2e344e;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 500;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .badge-tou { background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; }
    .badge-std { background-color: rgba(148, 163, 184, 0.2); color: #cbd5e1; }
    .recommendation-box {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        border-radius: 4px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .recommendation-item {
        color: #a7f3d0;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper to verify files exist
def check_pipeline_completed():
    required_files = [
        config.PROCESSED_DAILY_DATA_PATH,
        config.CLEANED_HOUSEHOLD_FEATURES_PATH,
        config.PROCESSED_DATA_DIR / "household_hourly_profiles.csv",
        config.MODELS_DIR / "consumption_forecaster.pkl",
        config.MODELS_DIR / "model_metadata.json",
        config.MODELS_DIR / "clustering_scaler.pkl",
        config.MODELS_DIR / "clustering_model.pkl"
    ]
    return all(os.path.exists(f) for f in required_files)

# Draw warning / initialization screen
def show_init_screen():
    st.markdown("<div class='main-header'>⚡ SmartGrid AI Energy Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Autonomous Smart Meter Forecasting & Tariff Optimization</div>", unsafe_allow_html=True)
    
    st.warning("⚠️ Pipeline files or trained models are missing. Please initialize the system first.")
    
    st.markdown("""
    ### How to Initialize the Project Pipeline:
    To build the database, train the machine learning forecaster and customer clustering models, please execute the pipeline stages sequentially.
    
    Run these commands in your project environment:
    ```powershell
    # 1. Preprocess daily smart meter data
    venv\\Scripts\\python -m src.data.make_dataset

    # 2. Run feature engineering (temporal, lag, weather interactions)
    venv\\Scripts\\python -m src.data.feature_engineering

    # 3. Train clustering model (categorizes consumer profiles)
    venv\\Scripts\\python -m src.models.train_clustering

    # 4. Process half-hourly data to construct hourly profiles
    venv\\Scripts\\python -m src.data.make_hourly_profiles

    # 5. Train AI consumption forecaster
    venv\\Scripts\\python -m src.models.train_forecaster
    ```
    """)
    
    # Check what files exist
    st.write("#### Pipeline Files Status:")
    status_data = []
    files_to_check = {
        "Daily Preprocessed Data": config.PROCESSED_DAILY_DATA_PATH,
        "Customer Clustering Profiles": config.CLEANED_HOUSEHOLD_FEATURES_PATH,
        "Half-Hourly Load Profiles": config.PROCESSED_DATA_DIR / "household_hourly_profiles.csv",
        "Forecaster Model Model (.pkl)": config.MODELS_DIR / "consumption_forecaster.pkl",
        "Forecaster Metadata (.json)": config.MODELS_DIR / "model_metadata.json"
    }
    for label, path in files_to_check.items():
        exists = os.path.exists(path)
        status_data.append({
            "Pipeline File": label,
            "Path": str(path),
            "Status": "✅ Available" if exists else "❌ Missing"
        })
    st.table(pd.DataFrame(status_data))

# Load data with caching
@st.cache_data
def load_all_data():
    daily_df = pd.read_csv(config.PROCESSED_DAILY_DATA_PATH)
    daily_df[config.DATE_COL] = pd.to_datetime(daily_df[config.DATE_COL])
    
    household_df = pd.read_csv(config.CLEANED_HOUSEHOLD_FEATURES_PATH)
    hourly_df = pd.read_csv(config.PROCESSED_DATA_DIR / "household_hourly_profiles.csv")
    
    with open(config.MODELS_DIR / "model_metadata.json", "r") as f:
        forecaster_meta = json.load(f)
        
    return daily_df, household_df, hourly_df, forecaster_meta

@st.cache_resource
def load_forecaster_model():
    return joblib.load(config.MODELS_DIR / "consumption_forecaster.pkl")

# Recursive 7-day forecasting logic
def forecast_next_7_days(model, hist_df, household_id, feature_cols, acorn_code, std_or_tou):
    # Get last 14 records of the household to bootstrap lag and rolling calculations
    hh_hist = hist_df[hist_df[config.HOUSEHOLD_COL] == household_id].sort_values(config.DATE_COL).tail(30).copy()
    if len(hh_hist) < 14:
        # Fallback if too short
        return None
        
    last_date = hh_hist[config.DATE_COL].max()
    
    # We require weather daily data to lookup forecasted daily weather
    weather_df = pd.read_csv(config.WEATHER_DAILY_PATH)
    weather_df["date_str"] = pd.to_datetime(weather_df["time"]).dt.strftime("%Y-%m-%d")
    weather_df["date_str"] = pd.to_datetime(weather_df["date_str"])
    
    # Get holidays
    holidays_df = pd.read_csv(config.HOLIDAYS_PATH)
    holidays_df["date_str"] = pd.to_datetime(holidays_df["Bank holidays"])
    holidays_set = set(holidays_df["date_str"].dt.date)
    
    # Prepare list for predictions
    predictions = []
    
    # Bootstrap data buffer (day, energy_sum)
    buffer = hh_hist[[config.DATE_COL, config.TARGET_COL]].to_dict("records")
    
    # Make forecasts recursively
    for step in range(1, 8):
        future_date = last_date + timedelta(days=step)
        
        # 1. Generate temporal features
        month = future_date.month
        day_of_week = future_date.dayofweek
        day_of_year = future_date.dayofyear
        is_weekend = int(day_of_week >= 5)
        season = (month%12 + 3)//3
        is_holiday = int(future_date.date() in holidays_set)
        
        # 2. Get weather parameters
        weather_match = weather_df[weather_df["date_str"] == future_date]
        if not weather_match.empty:
            temp_max = weather_match.iloc[0]["temperatureMax"]
            temp_min = weather_match.iloc[0]["temperatureMin"]
            wind_speed = weather_match.iloc[0]["windSpeed"]
            humidity = weather_match.iloc[0]["humidity"]
            visibility = weather_match.iloc[0]["visibility"]
            cloud_cover = weather_match.iloc[0]["cloudCover"]
        else:
            # Median fallbacks
            temp_max = 12.0
            temp_min = 6.0
            wind_speed = 3.0
            humidity = 0.8
            visibility = 10.0
            cloud_cover = 0.5
            
        temp_avg = (temp_max + temp_min) / 2
        heating_dd = max(0, 18.0 - temp_avg)
        cooling_dd = max(0, temp_avg - 18.0)
        
        # 3. Compute Lags (from the buffer)
        lags = {}
        for lag in config.LAG_DAYS:
            # Fetch index: -lag from buffer end
            if len(buffer) >= lag:
                lags[f"lag_{lag}"] = buffer[-lag][config.TARGET_COL]
            else:
                lags[f"lag_{lag}"] = buffer[-1][config.TARGET_COL] # fallback
                
        # 4. Compute Rolling Windows
        rolling = {}
        for w in config.ROLLING_WINDOWS:
            # Mean, Std, Max of the last w items in buffer
            last_w_vals = [x[config.TARGET_COL] for x in buffer[-w:]]
            rolling[f"rolling_mean_{w}"] = np.mean(last_w_vals)
            rolling[f"rolling_std_{w}"] = np.std(last_w_vals) if len(last_w_vals) > 1 else 0.0
            rolling[f"rolling_max_{w}"] = np.max(last_w_vals)
            
        # 5. Build feature dict
        feats = {
            "month": month,
            "day_of_week": day_of_week,
            "day_of_year": day_of_year,
            "is_weekend": is_weekend,
            "season": season,
            "is_holiday": is_holiday,
            "temperatureMax": temp_max,
            "temperatureMin": temp_min,
            "windSpeed": wind_speed,
            "humidity": humidity,
            "visibility": visibility,
            "cloudCover": cloud_cover,
            "heating_degree_days": heating_dd,
            "cooling_degree_days": cooling_dd,
            "stdorToU_encoded": 1 if std_or_tou == "ToU" else 0,
            "acorn_encoded": acorn_code
        }
        feats.update(lags)
        feats.update(rolling)
        
        # Filter feature columns exactly in order
        feature_vector = [feats.get(c, 0.0) for c in feature_cols]
        
        # Predict
        pred_val = float(model.predict([feature_vector])[0])
        pred_val = max(0.0, pred_val) # Clamp to zero
        
        predictions.append({
            config.DATE_COL: future_date,
            config.TARGET_COL: pred_val,
            "type": "Forecasted"
        })
        
        # Append prediction to buffer to feed subsequent recursive steps
        buffer.append({
            config.DATE_COL: future_date,
            config.TARGET_COL: pred_val
        })
        
    return pd.DataFrame(predictions)

# Main dashboard application
def main():
    if not check_pipeline_completed():
        show_init_screen()
        return
        
    # Load all processed assets
    daily_df, household_df, hourly_df, forecaster_meta = load_all_data()
    forecaster_model = load_forecaster_model()
    
    # Sidebar
    st.sidebar.markdown("### 🛠️ Navigation")
    app_mode = st.sidebar.radio(
        "Select Analytics Pane:",
        ["Grid Operator Overview", "Household Detail Analysis", "Tariff & Peak Shift Optimizer"]
    )
    
    # Side panel credits
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔋 SmartGrid AI Info")
    st.sidebar.info(
        "Using smart meter data (LCLids) from London to analyze "
        "energy load behavior, forecast demand using LightGBM, "
        "and optimize cost structures."
    )
    
    # Mode 1: Grid Overview
    if app_mode == "Grid Operator Overview":
        st.markdown("<div class='main-header'>⚡ Grid Operator Dashboard</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Consolidated Grid View, Cluster Profile Distributions & ML Performance</div>", unsafe_allow_html=True)
        
        # KPI Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{len(household_df)}</div>
                <div class='metric-label'>Households Monitored</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            avg_daily = daily_df[config.TARGET_COL].mean()
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{avg_daily:.2f} kWh</div>
                <div class='metric-label'>Avg Daily Household Usage</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            tou_count = (household_df["stdorToU"] == "ToU").sum()
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{tou_count}</div>
                <div class='metric-label'>Dynamic ToU Users</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            mape = forecaster_meta["metrics"].get("MAPE", 0.0)
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{mape:.2f}%</div>
                <div class='metric-label'>Model Accuracy (MAPE)</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### 📊 Customer Behavioral Segments")
        # Visualizing Clusters
        c_counts = household_df["usage_pattern"].value_counts().reset_index()
        c_counts.columns = ["Segment", "Count"]
        
        col_chart, col_explain = st.columns([1.2, 1])
        
        with col_chart:
            fig = px.pie(
                c_counts, 
                values="Count", 
                names="Segment", 
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title="Distribution of Grid Consumer Clusters"
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e6e8f0",
                legend=dict(orientation="h", y=-0.1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col_explain:
            st.markdown("#### Segment Characteristics:")
            st.markdown("""
            - 🟢 **Eco-Saver (Low Usage)**: Highly efficient consumers with flat, low usage. Low peak demand.
            - 🔵 **Standard Saver (Moderate Stable)**: Steady, moderate day-to-day consumption. Predictable load profile.
            - 🟡 **Comfort Consumer (Moderate Peaky)**: Moderate total usage, but concentrated peaks in evenings (16:00-20:00).
            - 🔴 **High Load (Heavy Consumer)**: Substantial energy demands. Heavy appliances, electric heating, or EVs present.
            """)
            # Show summary table
            clust_summary = household_df.groupby("usage_pattern")[config.CLUSTERING_FEATURES[:-1]].mean()
            clust_summary.columns = ["Avg (kWh)", "StDev (kWh)", "Peak Max (kWh)", "Min (kWh)"]
            st.dataframe(clust_summary.style.format("{:.2f}"))
            
    # Mode 2: Household Detail
    elif app_mode == "Household Detail Analysis":
        st.markdown("<div class='main-header'>🏠 Customer Energy Analyzer</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Analyze Household Historical Load Profiles, 7-Day Forecast & Key Drivers</div>", unsafe_allow_html=True)
        
        # Dropdown selection for household
        hh_list = sorted(household_df[config.HOUSEHOLD_COL].tolist())
        selected_hh = st.selectbox("Select Customer ID (LCLid):", hh_list)
        
        # Extract metadata
        meta = household_df[household_df[config.HOUSEHOLD_COL] == selected_hh].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"**Acorn Segment:** {meta['Acorn_grouped']}")
        with c2:
            st.markdown(f"**Tariff Type:** {meta['stdorToU']}")
        with c3:
            st.markdown(f"**Usage Profile:** {meta['usage_pattern']}")
        with c4:
            avg_usage = meta["mean_consumption"]
            st.markdown(f"**Daily Average:** {avg_usage:.2f} kWh")
            
        # Get historical daily consumption
        hh_daily = daily_df[daily_df[config.HOUSEHOLD_COL] == selected_hh].sort_values(config.DATE_COL).copy()
        
        # Forecast 7 days ahead
        # Get acorn integer mapping and stdorToU
        acorn_map = forecaster_meta["encoding_info"]["acorn_map"]
        acorn_code = acorn_map.get(meta['Acorn_grouped'], 0)
        
        forecast_df = forecast_next_7_days(
            forecaster_model, 
            daily_df, 
            selected_hh, 
            forecaster_meta["feature_cols"], 
            acorn_code, 
            meta["stdorToU"]
        )
        
        # Display line chart
        st.markdown("### 📈 Historical Consumption vs. AI 7-Day Forecast")
        if forecast_df is not None:
            # Combine history (last 21 days) and forecast
            hist_plot = hh_daily.tail(21)[[config.DATE_COL, config.TARGET_COL]].copy()
            hist_plot["type"] = "Historical"
            
            # Connect the gap between last hist and first forecast for continuous line
            last_hist_row = hist_plot.tail(1).copy()
            last_hist_row["type"] = "Forecasted"
            plot_df = pd.concat([hist_plot, last_hist_row, forecast_df], ignore_index=True)
            
            fig = px.line(
                plot_df, 
                x=config.DATE_COL, 
                y=config.TARGET_COL, 
                color="type",
                color_discrete_map={"Historical": "#3b82f6", "Forecasted": "#a855f7"},
                title=f"Energy Consumption Forecast for {selected_hh}"
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Daily Energy Sum (kWh)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e6e8f0"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Insufficient historical records to generate recursive forecast.")
            
        # Feature Importance Analysis
        st.markdown("### 🔑 Consumption Drivers (Feature Importance)")
        # Show general feature importances from forecaster model
        with open(config.MODELS_DIR / "model_metadata.json", "r") as f:
            meta_json = json.load(f)
            
        # Read from metrics if trained model has importance
        # LightGBM/RF importances
        model_obj = load_forecaster_model()
        if hasattr(model_obj, "feature_importances_"):
            importances = model_obj.feature_importances_
            feature_cols = meta_json["feature_cols"]
            imp_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(10)
            
            fig_imp = px.bar(
                imp_df, 
                x="Importance", 
                y="Feature", 
                orientation="h",
                color="Importance",
                color_continuous_scale=px.colors.sequential.Agsunset,
                title="Top 10 AI Model Decision Drivers"
            )
            fig_imp.update_layout(
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e6e8f0"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
    # Mode 3: Optimizer
    elif app_mode == "Tariff & Peak Shift Optimizer":
        st.markdown("<div class='main-header'>🔋 Tariff & Peak Shift Optimizer</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Interactive Tariff Comparison & Custom Load Shifting Simulation</div>", unsafe_allow_html=True)
        
        hh_list = sorted(household_df[config.HOUSEHOLD_COL].tolist())
        selected_hh = st.selectbox("Select Customer ID (LCLid):", hh_list)
        
        # Get household meta and hourly typical curve
        meta = household_df[household_df[config.HOUSEHOLD_COL] == selected_hh].iloc[0]
        
        # Load hourly profiles
        hourly_profiles_df = pd.read_csv(config.PROCESSED_DATA_DIR / "household_hourly_profiles.csv", index_col=config.HOUSEHOLD_COL)
        
        if selected_hh not in hourly_profiles_df.index:
            st.error("Hourly profile data is missing for this household.")
            return
            
        hh_profile = hourly_profiles_df.loc[selected_hh]
        
        # Display side-by-side KPI tariff comparison
        total_daily_kWh = hh_profile.sum()
        
        # Calculate standard flat cost
        flat_cost = energy_optimizer.calculate_flat_cost(total_daily_kWh)
        
        # Calculate ToU cost
        tou_details = energy_optimizer.calculate_tou_cost(hh_profile)
        tou_cost = tou_details["total_cost"]
        
        st.markdown("### 💷 Tariff Plan Cost Comparison")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>£{flat_cost:.2f}</div>
                <div class='metric-label'>Flat Rate Tariff (Daily)</div>
                <div class='status-badge badge-std'>Standard: 14.22p/kWh</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_c2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>£{tou_cost:.2f}</div>
                <div class='metric-label'>Time-of-Use (ToU) Tariff (Daily)</div>
                <div class='status-badge badge-tou'>Peak: 24.86p / OffPeak: 5.00p</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_c3:
            diff = flat_cost - tou_cost
            cheaper_plan = "Time-of-Use" if diff > 0 else "Flat Rate Standard"
            savings_pct = (abs(diff) / max(flat_cost, 0.001)) * 100
            
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>£{abs(diff):.2f}</div>
                <div class='metric-label'>Potential Savings ({savings_pct:.1f}%)</div>
                <div class='status-badge badge-tou' style='background-color:rgba(16,185,129,0.2);color:#10b981'>
                    Recommend: {cheaper_plan}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Slider for peak shifting simulation
        st.markdown("---")
        st.markdown("### 🛠️ Interactive Peak Load Shifting Simulation")
        st.write(
            "Time-of-Use tariffs charge a premium during **Peak hours (16:00 - 20:00)** "
            "but offer cheap rates during **Off-Peak hours (00:00 - 07:00)**. Use the slider below to simulate shifting "
            "heavy appliance loads (laundry, charging, dishwashing) out of the evening peak to night off-peak."
        )
        
        shift_percent = st.slider("Percentage of peak load to shift:", min_value=0, max_value=50, value=15, step=5)
        
        # Run simulation
        sim_results = energy_optimizer.simulate_peak_shift(hh_profile, shift_percent / 100.0)
        shifted_profile = sim_results["shifted_profile"]
        orig_tou_cost = sim_results["original_cost_details"]["total_cost"]
        new_tou_cost = sim_results["new_cost_details"]["total_cost"]
        savings = sim_results["savings"]
        shifted_energy = sim_results["shifted_energy_kWh"]
        
        # Plot both curves
        st.markdown("#### Hourly Load Curve Simulation")
        
        # Map indices to actual hour labels (e.g. 0 -> 00:00, 1 -> 00:30, etc.)
        times = []
        for idx in range(48):
            hour = idx // 2
            minute = "30" if idx % 2 != 0 else "00"
            times.append(f"{hour:02d}:{minute}")
            
        fig_curve = go.Figure()
        
        # Original profile line
        fig_curve.add_trace(go.Scatter(
            x=times, 
            y=hh_profile.values,
            mode='lines',
            name='Original Consumption Profile',
            line=dict(color='#3b82f6', width=2.5)
        ))
        
        # Shifted profile line
        fig_curve.add_trace(go.Scatter(
            x=times, 
            y=shifted_profile.values,
            mode='lines',
            name='Shifted/Simulated Profile',
            line=dict(color='#10b981', width=2.5, dash='dash')
        ))
        
        # Add Peak shaded region (indices 32 to 39, i.e. 16:00 to 20:00)
        fig_curve.add_vrect(
            x0="16:00", x1="20:00", 
            fillcolor="rgba(239, 68, 68, 0.08)", 
            layer="below", 
            line_width=0,
            annotation_text="Peak Rate (£0.2486)", 
            annotation_position="top left",
            annotation_font=dict(color="#ef4444", size=10)
        )
        
        # Add Off-peak shaded region (indices 0 to 13, i.e. 00:00 to 07:00)
        fig_curve.add_vrect(
            x0="00:00", x1="07:00", 
            fillcolor="rgba(16, 185, 129, 0.08)", 
            layer="below", 
            line_width=0,
            annotation_text="Off-Peak Rate (£0.0500)", 
            annotation_position="top left",
            annotation_font=dict(color="#10b981", size=10)
        )
        
        fig_curve.update_layout(
            xaxis_title="Time of Day",
            yaxis_title="Energy Consumption (kWh/hh)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6e8f0",
            legend=dict(orientation="h", y=1.1)
        )
        
        st.plotly_chart(fig_curve, use_container_width=True)
        
        # Savings summary
        st.markdown("#### Simulation Savings Summary:")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Shifted Energy Daily", f"{shifted_energy:.3f} kWh")
        with sc2:
            st.metric("New Daily ToU Cost", f"£{new_tou_cost:.2f}", delta=f"-£{savings:.2f}")
        with sc3:
            ann_savings = savings * 365
            st.metric("Estimated Annual Savings", f"£{ann_savings:.2f}")
            
        # Personalized Insights Section
        st.markdown("---")
        st.markdown("### 💡 Personalized Smart Grid Recommendations")
        rec_list = energy_optimizer.generate_efficiency_insights(
            sim_results["new_cost_details"], 
            meta["usage_pattern"]
        )
        
        st.markdown("<div class='recommendation-box'>", unsafe_allow_html=True)
        for item in rec_list:
            st.markdown(f"<div class='recommendation-item'>✔ {item}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

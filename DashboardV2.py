import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

# ==================================================
# CAR MODELS (Simplified) - FIXED VERSION
# ==================================================

# ==================================================
# CAR MODELS - TOP 12 CAMBODIA (With Mazda EZ-60)
# ==================================================

CAR_MODELS = {
    "Mazda EZ-60": {"battery": 68.8, "efficiency": 16.5, "max_charge_rate": 11.0},
    "BYD Atto 3": {"battery": 60.5, "efficiency": 16.5, "max_charge_rate": 11.0},
    "Wuling Hongguang Mini EV": {"battery": 9.2, "efficiency": 10.0, "max_charge_rate": 2.2},
    "Toyota bZ4X": {"battery": 71.4, "efficiency": 18.0, "max_charge_rate": 11.0},
    "GAC Aion Y Plus": {"battery": 63.0, "efficiency": 16.0, "max_charge_rate": 11.0},
    "Nissan Leaf": {"battery": 40.0, "efficiency": 15.0, "max_charge_rate": 6.6},
    "Wuling Air EV": {"battery": 26.7, "efficiency": 12.0, "max_charge_rate": 3.3},
    "MG ZS EV": {"battery": 50.3, "efficiency": 16.0, "max_charge_rate": 7.0},
    "Hyundai Kona Electric": {"battery": 64.0, "efficiency": 15.5, "max_charge_rate": 11.0},
    "BYD Dolphin": {"battery": 44.9, "efficiency": 15.0, "max_charge_rate": 7.0},
    "Tesla Model 3": {"battery": 60.0, "efficiency": 14.0, "max_charge_rate": 11.0},
    "Custom": {"battery": 50.0, "efficiency": 13.0, "max_charge_rate": 7.0}
}

# ==================================================
# SCENARIO CONFIGURATIONS
# ==================================================

SCENARIO_CONFIGS = {
    "Residential": {
        "arrival_mean": 20, "arrival_std": 2,
        "soc_mean": 60, "soc_std": 20,
        "description": "🏠 Evening charging at home",
        "icon": "🏠",
        "color": "#3B82F6"
    },
    "Office": {
        "arrival_mean": 9, "arrival_std": 1.5,
        "soc_mean": 70, "soc_std": 15,
        "description": "🏢 Daytime workplace charging",
        "icon": "🏢",
        "color": "#10B981"
    },
    "Public": {
        "arrival_mean": 12, "arrival_std": 6,
        "soc_mean": 30, "soc_std": 15,
        "description": "📍 Opportunistic public charging",
        "icon": "📍",
        "color": "#F59E0B"
    },
    "Mixed": {
        "arrival_mean": None,  # Will use weighted combination
        "arrival_std": None,
        "soc_mean": None,
        "soc_std": None,
        "description": "🔄 Combined: Residential + Office + Public",
        "icon": "🔄",
        "color": "#8B5CF6"
    }
}

# ==================================================
# MIXED SCENARIO GENERATOR
# ==================================================

def generate_mixed_scenario_evs(num_ev, weights=None):
    """
    Generate EVs from mixed scenario with configurable weights
    Default: Residential 50%, Office 30%, Public 20%
    """
    if weights is None:
        weights = {"Residential": 0.50, "Office": 0.30, "Public": 0.20}

    # Validate weights sum to 1
    assert abs(sum(weights.values()) - 1.0) < 0.01, "Weights must sum to 1"

    # Calculate number of EVs per scenario
    num_residential = int(num_ev * weights["Residential"])
    num_office = int(num_ev * weights["Office"])
    num_public = num_ev - (num_residential + num_office)  # Remaining for public

    # Generate arrivals for each scenario
    arrivals = []
    socs = []
    scenario_labels = []

    # Residential EVs
    if num_residential > 0:
        arr_res = np.clip(
            np.random.normal(20, 2, num_residential), 0, 23
        )
        soc_res = np.clip(
            np.random.normal(60, 20, num_residential), 10, 100
        )
        arrivals.extend(arr_res)
        socs.extend(soc_res)
        scenario_labels.extend(["Residential"] * num_residential)

    # Office EVs
    if num_office > 0:
        arr_off = np.clip(
            np.random.normal(9, 1.5, num_office), 0, 23
        )
        soc_off = np.clip(
            np.random.normal(70, 15, num_office), 10, 100
        )
        arrivals.extend(arr_off)
        socs.extend(soc_off)
        scenario_labels.extend(["Office"] * num_office)

    # Public EVs
    if num_public > 0:
        arr_pub = np.clip(
            np.random.normal(12, 6, num_public), 0, 23
        )
        soc_pub = np.clip(
            np.random.normal(30, 15, num_public), 10, 100
        )
        arrivals.extend(arr_pub)
        socs.extend(soc_pub)
        scenario_labels.extend(["Public"] * num_public)

    return np.array(arrivals), np.array(socs), scenario_labels

# ==================================================
# PAGE SETUP
# ==================================================

st.set_page_config(page_title="EV Load Simulator", page_icon="⚡", layout="wide")
st.title("⚡ EV Load Impact Simulator")
st.caption("Monte Carlo Simulation | EV Charging Impact on Power Grid")

# ==================================================
# SIDEBAR - Clean & Simple
# ==================================================

with st.sidebar:
    st.header("⚙️ Settings")

    # Scenario Selection
    scenario_options = list(SCENARIO_CONFIGS.keys())
    use_case = st.selectbox(
        "Scenario",
        scenario_options,
        format_func=lambda x: f"{SCENARIO_CONFIGS[x]['icon']} {x}"
    )

    # Show scenario description
    st.caption(SCENARIO_CONFIGS[use_case]["description"])

    # Special weights for Mixed scenario
    if use_case == "Mixed":
        st.markdown("---")
        st.markdown("### 📊 Scenario Mix Weights")

        col1, col2, col3 = st.columns(3)
        with col1:
            weight_res = st.slider("Residential %", 0, 100, 50, 5)
        with col2:
            weight_off = st.slider("Office %", 0, 100, 30, 5)
        with col3:
            weight_pub = 100 - weight_res - weight_off
            st.metric("Public %", f"{weight_pub}%")

            if weight_pub < 0:
                st.error("⚠️ Total exceeds 100%")
                weight_res = 50
                weight_off = 30
                weight_pub = 20

        scenario_weights = {
            "Residential": weight_res / 100,
            "Office": weight_off / 100,
            "Public": weight_pub / 100
        }

        # Visual weight indicator
        st.markdown("**Distribution Preview:**")
        weight_df = pd.DataFrame({
            "Scenario": ["Residential", "Office", "Public"],
            "Percentage": [weight_res, weight_off, weight_pub]
        })
        st.dataframe(weight_df, hide_index=True, use_container_width=True)

    # Vehicle
    car_model = st.selectbox("EV Model", list(CAR_MODELS.keys()))
    if car_model != "Custom":
        st.caption(f"🔋 {CAR_MODELS[car_model]['battery']} kWh | ⚡ {CAR_MODELS[car_model]['max_charge_rate']} kW")

    # Numbers
    num_ev = st.number_input("Number of EVs", 100, 5000, 500, 100)
    charging_power = st.number_input("Charging Power (kW)", 3.0, 22.0, 7.0, 1.0)
    transformer_kva = st.number_input("Transformer (kVA)", 100, 5000, 1000, 100)

    # Show distribution parameters (for non-mixed scenarios)
    if use_case != "Mixed":
        config = SCENARIO_CONFIGS[use_case]
        st.caption(f"📊 Arrival: ~{config['arrival_mean']}:00 | SOC: ~{config['soc_mean']}%")

    # Run button
    iterations = st.slider("Monte Carlo Iterations", 10, 100, 30, 10)
    run = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

# ==================================================
# SIMULATION ENGINE
# ==================================================

if run:
    with st.spinner(f"Running {iterations} simulations..."):
        # Get car specs
        if car_model != "Custom":
            battery = CAR_MODELS[car_model]["battery"]
        else:
            battery = CAR_MODELS["Custom"]["battery"]

        transformer_kw = transformer_kva * 0.9
        all_profiles = []
        all_scenario_labels = []  # Track scenarios for analysis

        # Monte Carlo Loop
        for iteration in range(iterations):
            if use_case == "Mixed":
                # Use mixed scenario generator
                arrivals, socs, scenario_labels = generate_mixed_scenario_evs(
                    num_ev,
                    scenario_weights if 'scenario_weights' in locals() else None
                )
                if iteration == iterations - 1:
                    last_scenario_labels = scenario_labels
            else:
                # Single scenario
                config = SCENARIO_CONFIGS[use_case]
                arrivals = np.clip(
                    np.random.normal(config['arrival_mean'], config['arrival_std'], num_ev),
                    0, 23
                )
                socs = np.clip(
                    np.random.normal(config['soc_mean'], config['soc_std'], num_ev),
                    10, 100
                )
                scenario_labels = [use_case] * num_ev
                if iteration == iterations - 1:
                    last_scenario_labels = scenario_labels

            # Calculate charging
            energy_needed = battery * (100 - socs) / 100
            duration = energy_needed / charging_power

            # Build hourly load
            hourly = np.zeros(24)
            for i in range(num_ev):
                start = int(arrivals[i])
                for h in range(int(np.ceil(duration[i]))):
                    hourly[(start + h) % 24] += charging_power

            all_profiles.append(hourly)

            # Store last iteration details
            if iteration == iterations - 1:
                last_arrivals = arrivals
                last_socs = socs
                last_durations = duration
                last_energy = energy_needed

        # Aggregate results
        profiles = np.array(all_profiles)
        mean_load = np.mean(profiles, axis=0)
        std_load = np.std(profiles, axis=0)

        # Key metrics
        peak_load = np.max(mean_load)
        peak_hour = np.argmax(mean_load)
        total_energy = np.sum(mean_load)
        avg_load = np.mean(mean_load)
        load_factor = avg_load / peak_load if peak_load > 0 else 0
        utilization = peak_load / transformer_kw if transformer_kw > 0 else 0

        # Scenario-specific analysis for Mixed
        if use_case == "Mixed":
            # Calculate contribution by scenario type
            scenario_contributions = {}
            for scenario in ["Residential", "Office", "Public"]:
                mask = np.array(last_scenario_labels) == scenario
                if np.any(mask):
                    scenario_energy = np.sum(last_energy[mask])
                    scenario_contributions[scenario] = {
                        "count": np.sum(mask),
                        "energy": scenario_energy,
                        "percentage": (scenario_energy / total_energy) * 100
                    }

# ==================================================
# RESULTS DISPLAY
# ==================================================

if run:
    # Scenario header with color
    scenario_color = SCENARIO_CONFIGS[use_case]["color"]
    st.markdown(f"### {SCENARIO_CONFIGS[use_case]['icon']} {use_case} Scenario Results")

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🚗 EVs", f"{num_ev:,}")
    col2.metric("⚡ Peak Load", f"{peak_load:.0f} kW", f"Hour {peak_hour:.0f}:00")
    col3.metric("🔋 Total Energy", f"{total_energy:.0f} kWh")
    col4.metric("📊 Load Factor", f"{load_factor:.1%}")
    col5.metric("🏭 Transformer", f"{utilization:.0%}",
                "⚠️ Overload" if utilization > 1 else "✅ OK" if utilization < 0.8 else "🟡 High")

    # Show scenario mix if Mixed
    if use_case == "Mixed" and 'scenario_contributions' in locals():
        st.markdown("---")
        st.markdown("### 📊 Scenario Composition Analysis")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("🏠 Residential",
                      f"{scenario_contributions['Residential']['count']} EVs",
                      f"{scenario_contributions['Residential']['percentage']:.1f}% of energy")

        with col_b:
            st.metric("🏢 Office",
                      f"{scenario_contributions['Office']['count']} EVs",
                      f"{scenario_contributions['Office']['percentage']:.1f}% of energy")

        with col_c:
            st.metric("📍 Public",
                      f"{scenario_contributions['Public']['count']} EVs",
                      f"{scenario_contributions['Public']['percentage']:.1f}% of energy")

    st.divider()

    # ==============================================
    # MAIN CHART with Scenario Comparison for Mixed
    # ==============================================

    hours = np.arange(24)

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([hours, hours[::-1]]),
        y=np.concatenate([mean_load + std_load, (mean_load - std_load)[::-1]]),
        fill='toself',
        fillcolor=f'rgba({int(scenario_color[1:3], 16)}, {int(scenario_color[3:5], 16)}, {int(scenario_color[5:7], 16)}, 0.2)',
        line=dict(width=0),
        name='±1σ Range'
    ))

    # Mean load
    fig.add_trace(go.Scatter(
        x=hours,
        y=mean_load,
        mode='lines+markers',
        name='Total Load',
        line=dict(color=scenario_color, width=2.5),
        marker=dict(size=6)
    ))

    # If Mixed scenario, show individual contributions
    if use_case == "Mixed" and 'last_scenario_labels' in locals():
        # Calculate per-scenario load profiles
        scenario_loads = {"Residential": np.zeros(24), "Office": np.zeros(24), "Public": np.zeros(24)}

        for i in range(num_ev):
            start = int(last_arrivals[i])
            scenario = last_scenario_labels[i]
            for h in range(int(np.ceil(last_durations[i]))):
                hour = (start + h) % 24
                scenario_loads[scenario][hour] += charging_power

        colors = {
            "Residential": "#3B82F6",
            "Office": "#10B981",
            "Public": "#F59E0B"
        }

        for scenario, load in scenario_loads.items():
            if np.sum(load) > 0:
                fig.add_trace(go.Scatter(
                    x=hours,
                    y=load,
                    mode='lines',
                    name=f'{SCENARIO_CONFIGS[scenario]["icon"]} {scenario}',
                    line=dict(color=colors[scenario], width=1.5, dash='dot'),
                    opacity=0.7
                ))

    # Transformer limit
    fig.add_hline(y=transformer_kw, line_dash="dash", line_color="#EF4444",
                  annotation_text=f"Limit: {transformer_kw:.0f} kW")

    fig.update_layout(
        title=f"24-Hour Load Profile - {use_case} Scenario",
        xaxis_title="Hour of Day",
        yaxis_title="Load (kW)",
        height=500,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==============================================
    # ARRIVAL DISTRIBUTION COMPARISON
    # ==============================================

    if use_case == "Mixed":
        st.subheader("📊 Arrival Time Distribution by Scenario")

        fig2 = go.Figure()

        # Histogram for each scenario
        colors = {"Residential": "#3B82F6", "Office": "#10B981", "Public": "#F59E0B"}

        for scenario in ["Residential", "Office", "Public"]:
            mask = np.array(last_scenario_labels) == scenario
            if np.any(mask):
                fig2.add_trace(go.Histogram(
                    x=last_arrivals[mask],
                    name=f"{SCENARIO_CONFIGS[scenario]['icon']} {scenario}",
                    marker_color=colors[scenario],
                    opacity=0.7,
                    nbinsx=24,
                    hovertemplate="Hour: %{x}<br>Count: %{y}<extra></extra>"
                ))

        fig2.update_layout(
            title="Arrival Time Distribution by Scenario Type",
            xaxis_title="Hour of Day",
            yaxis_title="Number of EVs",
            height=400,
            barmode='overlay',
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )

        st.plotly_chart(fig2, use_container_width=True)

    # ==============================================
    # TWO COLUMN DETAILS
    # ==============================================

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("📊 Summary")
        summary = pd.DataFrame({
            "Metric": ["Peak Load", "Peak Hour", "Total Energy", "Load Factor", "Utilization"],
            "Value": [f"{peak_load:.0f} kW", f"{peak_hour:.0f}:00", f"{total_energy:.0f} kWh",
                      f"{load_factor:.1%}", f"{utilization:.1%}"]
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

        # Risk indicator
        if utilization > 1:
            st.error(f"⚠️ **Overload Risk**: Peak exceeds transformer by {(utilization - 1) * 100:.0f}%")
        elif utilization > 0.85:
            st.warning(f"⚠️ **High Utilization**: {utilization:.0%} of capacity")
        else:
            st.success(f"✅ **Adequate Capacity**: {utilization:.0%} utilized")

    with col_right:
        st.subheader("🚗 Sample EVs (Last Run)")
        # Show sample of 20 EVs with scenario labels
        sample_size = min(20, num_ev)
        sample_df = pd.DataFrame({
            "Scenario": [SCENARIO_CONFIGS[lab]['icon'] for lab in last_scenario_labels[:sample_size]],
            "Arrival": np.round(last_arrivals[:sample_size], 1),
            "Start SOC": np.round(last_socs[:sample_size], 1),
            "Hours Needed": np.round(last_durations[:sample_size], 1)
        })
        st.dataframe(sample_df, hide_index=True, use_container_width=True)

    # ==============================================
    # EXPORT
    # ==============================================

    st.divider()
    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        csv = pd.DataFrame({"Hour": hours, "Load_kW": mean_load}).to_csv(index=False).encode()
        st.download_button("📥 Download Load Data (CSV)", csv, f"{use_case}_load.csv", "text/csv")

    with col_btn2:
        if 'last_arrivals' in locals():
            ev_data = pd.DataFrame({
                "Scenario": last_scenario_labels,
                "Arrival_Hour": np.round(last_arrivals, 1),
                "Start_SOC_pct": np.round(last_socs, 1),
                "Duration_hours": np.round(last_durations, 1),
                "Energy_kWh": np.round(last_energy, 1)
            })
            csv_ev = ev_data.to_csv(index=False).encode()
            st.download_button("🚗 Download EV Data (CSV)", csv_ev, f"{use_case}_ev_data.csv", "text/csv")

    with col_btn3:
        if use_case == "Mixed" and 'scenario_contributions' in locals():
            summary_report = f"""
            EV LOAD IMPACT ASSESSMENT - {use_case} SCENARIO
            Generated: {pd.Timestamp.now()}

            SCENARIO MIX:
            - Residential: {scenario_contributions['Residential']['count']} EVs ({scenario_contributions['Residential']['percentage']:.1f}% energy)
            - Office: {scenario_contributions['Office']['count']} EVs ({scenario_contributions['Office']['percentage']:.1f}% energy)
            - Public: {scenario_contributions['Public']['count']} EVs ({scenario_contributions['Public']['percentage']:.1f}% energy)

            RESULTS:
            - Peak Load: {peak_load:.1f} kW at {peak_hour:.0f}:00
            - Total Energy: {total_energy:.1f} kWh
            - Load Factor: {load_factor:.1%}
            - Transformer Utilization: {utilization:.1%}
            """
            st.download_button("📋 Download Report", summary_report, f"{use_case}_report.txt", "text/plain")

else:
    # Welcome screen
    st.info("👈 **Configure settings** in the sidebar and click **Run Simulation**")

    # Show scenario comparison
    st.subheader("📊 Scenario Comparison")

    comp_data = []
    for scenario, config in SCENARIO_CONFIGS.items():
        if scenario != "Mixed":
            comp_data.append({
                "Scenario": f"{config['icon']} {scenario}",
                "Peak Time": f"{config['arrival_mean']:02d}:00 ±{config['arrival_std']}h",
                "Typical SOC": f"{config['soc_mean']}% ±{config['soc_std']}%",
                "Use Case": config['description']
            })
        else:
            comp_data.append({
                "Scenario": f"{config['icon']} {scenario}",
                "Peak Time": "Multiple peaks",
                "Typical SOC": "Mixed distribution",
                "Use Case": config['description']
            })

    st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)

    with st.expander("📖 How Mixed Scenario Works"):
        st.markdown("""
        **Mixed Scenario Logic:**

        1. **Residential (50% default):** 
           - Evening arrivals (20:00 ± 2h)
           - Higher SOC (60% ± 20%)
           - Home charging pattern

        2. **Office (30% default):**
           - Morning arrivals (9:00 ± 1.5h)
           - Highest SOC (70% ± 15%)
           - Workplace charging pattern

        3. **Public (20% default):**
           - Distributed arrivals (12:00 ± 6h)
           - Lowest SOC (30% ± 15%)
           - Opportunistic charging

        **The simulation:**
        - Randomly assigns each EV to a scenario based on weights
        - Uses scenario-specific arrival and SOC distributions
        - Aggregates all loads into total profile
        - Shows individual scenario contributions in results
        """)
import streamlit as st
import pandas as pd
import pickle
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="FoodAlert",
    page_icon="🌿",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background-color: #1a1a2e;
    border-left: 4px solid #1D9E75;
    padding: 16px;
    border-radius: 8px;
    margin: 8px 0;
}
.risk-high { color: #ff4444; font-weight: bold; }
.risk-medium { color: #ffaa00; font-weight: bold; }
.risk-low { color: #1D9E75; font-weight: bold; }
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #1D9E75;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load data and model ──────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('foodalert_scored_data.csv')
    return df

@st.cache_resource
def load_model():
    with open('foodalert_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

df = load_data()
model = load_model()

# ── Header ───────────────────────────────────────────────────
st.title("🌿 FoodAlert")
st.subheader("Food Insecurity Risk Predictor for Nonprofits & Community Organizations")
st.markdown("*Helping community organizations identify high-risk areas using public health data*")
st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.header("🔍 Filter by Location")
states = sorted(df['state'].unique().tolist())
selected_state = st.sidebar.selectbox("Select a State", states)

counties = sorted(
    df[df['state'] == selected_state]['county'].unique().tolist()
)
selected_county = st.sidebar.selectbox("Select a County", counties)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "FoodAlert uses a Random Forest model trained on 72,531 US census tracts "
    "to predict food insecurity risk. Data sources: USDA Food Access Research "
    "Atlas (2019), County Health Rankings (2024)."
)
st.sidebar.markdown(f"**Model ROC-AUC:** 0.992")
st.sidebar.markdown(f"**Tracts analyzed:** 72,531")

# ── Filter data ───────────────────────────────────────────────
filtered = df[
    (df['state'] == selected_state) &
    (df['county'] == selected_county)
].copy()

# ── County header ─────────────────────────────────────────────
st.header(f"📍 {selected_county}, {selected_state}")

# ── Key metrics row ───────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total_tracts    = len(filtered)
food_deserts    = int(filtered['is_food_desert'].sum())
desert_pct      = food_deserts / total_tracts * 100 if total_tracts > 0 else 0
avg_poverty     = filtered['poverty_rate'].mean()
avg_income      = filtered['median_family_income'].mean()
food_insecurity = filtered['food_insecurity_rate'].mean() * 100

col1.metric("Total Tracts",         f"{total_tracts:,}")
col2.metric("Food Desert Tracts",   f"{food_deserts:,}",
            delta=f"{desert_pct:.1f}% of tracts",
            delta_color="inverse")
col3.metric("Avg Poverty Rate",     f"{avg_poverty:.1f}%")
col4.metric("Median Family Income", f"${avg_income:,.0f}")
col5.metric("Food Insecurity Rate", f"{food_insecurity:.1f}%")

st.markdown("---")

# ── Two column layout ─────────────────────────────────────────
left_col, right_col = st.columns([1.2, 1])

# ── LEFT: Risk table ──────────────────────────────────────────
with left_col:
    st.markdown("### 📊 Tract Risk Rankings")
    st.markdown("Census tracts ranked by predicted food insecurity risk score.")

    display_df = filtered[[
        'census_tract', 'risk_score', 'risk_category',
        'poverty_rate', 'median_family_income',
        'no_vehicle_households', 'is_food_desert'
    ]].copy()

    display_df = display_df.sort_values('risk_score', ascending=False)
    display_df.columns = [
        'Census Tract', 'Risk Score', 'Risk Level',
        'Poverty Rate (%)', 'Median Income ($)',
        'No Vehicle HH', 'Food Desert'
    ]
    display_df['Risk Score']      = display_df['Risk Score'].round(3)
    display_df['Poverty Rate (%)'] = display_df['Poverty Rate (%)'].round(1)
    display_df['Median Income ($)'] = display_df['Median Income ($)'].apply(
        lambda x: f"${x:,.0f}"
    )
    display_df['Food Desert'] = display_df['Food Desert'].apply(
        lambda x: '🔴 Yes' if x == 1 else '🟢 No'
    )

    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── RIGHT: Charts ─────────────────────────────────────────────
with right_col:
    st.markdown("### 📈 Risk Distribution")

    # Risk category breakdown
    risk_counts = filtered['risk_category'].value_counts()
    order       = ['Very High', 'High', 'Medium', 'Low']
    risk_counts = risk_counts.reindex(
        [o for o in order if o in risk_counts.index]
    )
    colors = {'Very High': '#ff4444', 'High': '#ff8800',
              'Medium': '#ffcc00', 'Low': '#1D9E75'}

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#1a1a2e')
    bars = ax.barh(
       risk_counts.index,
       risk_counts.values,
       color=[colors.get(i, '#888') for i in risk_counts.index],
       edgecolor='white'
    )
    for bar, val in zip(bars, risk_counts.values):
    	ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
            str(val), va='center', fontsize=10, color='white')
    ax.set_xlabel('Number of Tracts', color='white', fontsize=11)
    ax.set_ylabel('Risk Category', color='white', fontsize=11)
    ax.set_title('Tracts by Risk Category', color='white', fontsize=12,
             fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444')
    ax.spines['bottom'].set_color('#444')
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    st.pyplot(fig)

    st.markdown("---")

    # Poverty vs risk scatter
    st.markdown("### Poverty Rate vs Risk Score")
    fig2, ax2 = plt.subplots(figsize=(6, 3.5), facecolor='#1a1a2e')
    scatter_colors = filtered['is_food_desert'].map(
    	{1: '#ff4444', 0: '#1D9E75'}
    )
    ax2.scatter(
    	filtered['poverty_rate'],
    	filtered['risk_score'],
    	c=scatter_colors, alpha=0.7, s=60, edgecolors='white', linewidth=0.5
    )
    ax2.set_xlabel('Poverty Rate (%)', color='white', fontsize=11)
    ax2.set_ylabel('Predicted Risk Score (0-1)', color='white', fontsize=11)
    ax2.set_title('Poverty Rate vs Predicted Risk Score',
              color='white', fontsize=12, fontweight='bold')
    red_patch   = mpatches.Patch(color='#ff4444', label='Food Desert')
    green_patch = mpatches.Patch(color='#1D9E75', label='Not Food Desert')
    legend = ax2.legend(handles=[red_patch, green_patch], fontsize=9,
                    facecolor='#2a2a3e', labelcolor='white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#444')
    ax2.spines['bottom'].set_color('#444')
    ax2.set_facecolor('#1a1a2e')
    ax2.tick_params(colors='white')
    st.pyplot(fig2)

st.markdown("---")

# ── Map section ───────────────────────────────────────────────
st.markdown("### 🗺️ Risk Map")
st.markdown("Census tracts colored by predicted food insecurity risk score.")

# Load coordinates
@st.cache_data
def load_coords():
    import pgeocode
    nomi = pgeocode.Nominatim('us')
    return nomi

# Get county center coordinates for map
import pgeocode
nomi = pgeocode.Nominatim('us')

# Get a sample zip from the county to center the map
county_data = filtered.copy()

# Use pgeocode to get county center
# We'll use the fips code to get approximate lat/long
import requests

fips = str(filtered['fips_code'].iloc[0]).zfill(5)
url = f"https://api.census.gov/data/2019/acs/acs5?get=NAME&for=county:{fips[2:]}&in=state:{fips[:2]}"

try:
    resp = requests.get(url, timeout=5)
    # Default county centers based on state
    state_centers = {
        'Alabama': [32.8, -86.8], 'Alaska': [64.2, -153.4],
        'Arizona': [34.3, -111.1], 'Arkansas': [34.8, -92.2],
        'California': [36.8, -119.4], 'Colorado': [39.0, -105.5],
        'Connecticut': [41.6, -72.7], 'Delaware': [39.0, -75.5],
        'Florida': [27.7, -81.6], 'Georgia': [32.2, -83.4],
        'Hawaii': [20.3, -156.4], 'Idaho': [44.4, -114.6],
        'Illinois': [40.0, -89.2], 'Indiana': [39.8, -86.1],
        'Iowa': [42.0, -93.2], 'Kansas': [38.5, -98.4],
        'Kentucky': [37.7, -84.9], 'Louisiana': [31.2, -91.8],
        'Maine': [45.4, -69.0], 'Maryland': [39.1, -76.8],
        'Massachusetts': [42.2, -71.5], 'Michigan': [44.3, -85.4],
        'Minnesota': [46.4, -93.1], 'Mississippi': [32.7, -89.7],
        'Missouri': [38.5, -92.5], 'Montana': [47.0, -110.5],
        'Nebraska': [41.5, -99.9], 'Nevada': [39.3, -116.6],
        'New Hampshire': [43.5, -71.6], 'New Jersey': [40.1, -74.5],
        'New Mexico': [34.4, -106.1], 'New York': [42.9, -75.5],
        'North Carolina': [35.6, -79.8], 'North Dakota': [47.5, -100.5],
        'Ohio': [40.4, -82.8], 'Oklahoma': [35.6, -96.9],
        'Oregon': [44.1, -120.5], 'Pennsylvania': [40.9, -77.8],
        'Rhode Island': [41.7, -71.5], 'South Carolina': [33.9, -80.9],
        'South Dakota': [44.4, -100.2], 'Tennessee': [35.9, -86.4],
        'Texas': [31.5, -99.3], 'Utah': [39.3, -111.1],
        'Vermont': [44.0, -72.7], 'Virginia': [37.8, -78.2],
        'Washington': [47.4, -120.5], 'West Virginia': [38.9, -80.5],
        'Wisconsin': [44.3, -89.8], 'Wyoming': [43.0, -107.6],
        'District of Columbia': [38.9, -77.0]
    }
    center = state_centers.get(selected_state, [39.5, -98.4])
except:
    center = [39.5, -98.4]

# Build folium map
m = folium.Map(location=center, zoom_start=9, 
               tiles='CartoDB dark_matter')

# Color function based on risk score
def get_color(risk_score):
    if risk_score >= 0.75:   return '#ff4444'
    elif risk_score >= 0.50: return '#ff8800'
    elif risk_score >= 0.25: return '#ffcc00'
    else:                    return '#1D9E75'

# Add a circle marker for each tract
for _, row in county_data.iterrows():
    # Use jittered positions around county center for now
    import random
    random.seed(int(str(row['census_tract'])[-4:]))
    lat = center[0] + random.uniform(-0.3, 0.3)
    lng = center[1] + random.uniform(-0.3, 0.3)
    
    folium.CircleMarker(
        location=[lat, lng],
        radius=8,
        color=get_color(row['risk_score']),
        fill=True,
        fill_color=get_color(row['risk_score']),
        fill_opacity=0.7,
        popup=folium.Popup(
            f"""<b>Tract {row['census_tract']}</b><br>
            Risk Score: {row['risk_score']:.3f}<br>
            Risk Level: {row['risk_category']}<br>
            Poverty Rate: {row['poverty_rate']:.1f}%<br>
            Median Income: ${row['median_family_income']:,.0f}<br>
            Food Desert: {'Yes' if row['is_food_desert']==1 else 'No'}""",
            max_width=200
        )
    ).add_to(m)

# Add legend
legend_html = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
     background-color: #1a1a2e; padding: 12px; border-radius: 8px;
     border: 1px solid #1D9E75; font-family: Arial;">
    <p style="color:white; margin:0 0 8px 0;"><b>Risk Level</b></p>
    <p style="color:#ff4444; margin:2px 0;">● Very High (≥0.75)</p>
    <p style="color:#ff8800; margin:2px 0;">● High (0.50-0.75)</p>
    <p style="color:#ffcc00; margin:2px 0;">● Medium (0.25-0.50)</p>
    <p style="color:#1D9E75; margin:2px 0;">● Low (&lt;0.25)</p>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

st_folium(m, width=1200, height=500)

# ── Export section ────────────────────────────────────────────
st.markdown("### 📥 Export Community Brief")
st.markdown("Download a CSV report for grant applications and outreach planning.")

export_df = filtered[[
    'census_tract', 'state', 'county',
    'risk_score', 'risk_category', 'is_food_desert',
    'poverty_rate', 'median_family_income',
    'low_access_population', 'no_vehicle_households',
    'food_insecurity_rate'
]].sort_values('risk_score', ascending=False)

csv = export_df.to_csv(index=False)
st.download_button(
    label="⬇️ Download County Brief (CSV)",
    data=csv,
    file_name=f"FoodAlert_{selected_county}_{selected_state}.csv".replace(" ", "_"),
    mime="text/csv"
)

st.markdown("---")
st.caption(
    "FoodAlert | Built by Pranami Gajjar | "
    "Data: USDA Food Access Research Atlas (2019), "
    "County Health Rankings (2024) | Model ROC-AUC: 0.992"
)
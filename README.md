# 🌿 FoodAlert — Food Insecurity Risk Predictor

**A data-driven tool that helps nonprofits and community organizations identify 
high-risk food insecurity areas across the United States.**

🔗 **[Live App](https://foodalert-3kgp7eypopvlfuv84sfmoz.streamlit.app/)** | 
Built by [Pranami Gajjar](https://linkedin.com/in/pranami-gajjar045)

---

## The Problem

Food insecurity affects over 44 million Americans, yet the organizations best 
positioned to address it — local food banks, nonprofits, and community advocates 
— operate without reliable data tools. Public datasets from the USDA and Census 
Bureau contain rich, actionable information, but they are locked behind complex 
portals that require data expertise to navigate.

The result: resource allocation decisions are made on intuition rather than 
evidence. High-risk neighborhoods get missed. Grant applications lack quantitative 
backing. Outreach misses the communities that need it most.

---

## The Solution

FoodAlert translates public health and census data into a plain-language, 
interactive risk map — **no data expertise required.** A nonprofit worker selects 
their state and county and within seconds sees:

- Which census tracts carry the highest food insecurity risk
- How risk is distributed across their service area
- Key drivers: poverty rate, income, grocery access, vehicle availability
- An exportable CSV brief for grant applications and outreach planning

---

## Key Findings

- **12.8% of all US census tracts** (9,293 tracts) are classified as food deserts
- Food desert tracts have **10.6 percentage points higher poverty** than non-food-desert tracts
- Families in non-food-desert areas earn **$31,879 more per year** on average
- Low access population is **2.5x higher** in food desert tracts
- Food deserts are **not just a rural problem** — urban tracts are equally affected
- The model shows **consistent performance across urban and rural geographies** 
  (recall: 0.975 urban vs 0.967 rural) — a key fairness finding

---

## Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.992 |
| Overall Accuracy | 97% |
| Food Desert Recall | 87% |
| Food Desert Precision | 93% |

**Algorithm:** Random Forest (100 trees, class_weight='balanced')  
**Training data:** 58,024 census tracts  
**Test data:** 14,507 census tracts

### Fairness Audit

| Group | Food Desert Recall |
|-------|--------------------|
| Urban tracts | 0.975 |
| Rural tracts | 0.967 |
| Low poverty (0-10%) | 0.849 |
| Medium poverty (10-20%) | 0.954 |
| High poverty (20-40%) | 0.999 |
| Very high poverty (40%+) | 0.998 |

The model performs equitably across urban and rural geographies. Slightly lower 
recall in low-poverty tracts suggests geographic access barriers in affluent areas 
may be undercaptured — a direction for future improvement.

---

## Data Sources

| Dataset | Provider | Variables Used |
|---------|----------|----------------|
| Food Access Research Atlas (2019) | USDA Economic Research Service | Food desert flags, store distance, vehicle access |
| American Community Survey (5-yr) | US Census Bureau | Poverty rate, median income, population |
| County Health Rankings (2024) | University of Wisconsin | Food insecurity %, health outcomes |

All data sources are free and publicly available.

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Development | Google Colab |
| Data processing | Python, Pandas |
| Machine learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Mapping | Folium |
| Web application | Streamlit |
| Deployment | Streamlit Community Cloud |
| Version control | GitHub |

**Total cost: $0**

---

## How to Run Locally
```bash
# Clone the repo
git clone https://github.com/pranami4501/foodalert.git
cd foodalert

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Project Structure
```
foodalert/
├── app.py                      # Streamlit application
├── foodalert_scored_data.csv   # 72,531 tracts with risk scores
├── foodalert_model.pkl         # Trained Random Forest model
├── foodalert_model_info.pkl    # Model metadata
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## Methodology

1. **Data Collection** — Downloaded USDA Food Access Research Atlas, Census ACS, 
   and County Health Rankings data. Merged on FIPS county codes.
2. **Feature Engineering** — Selected 7 features based on domain knowledge and 
   VariableLookup documentation: poverty rate, median family income, low access 
   population, no-vehicle households, food insecurity rate, urban flag, population.
3. **Modeling** — Trained Random Forest classifier with class_weight='balanced' 
   to handle 87/13 class imbalance. Evaluated on held-out 20% test set.
4. **Fairness Audit** — Measured recall across urban/rural and poverty level 
   subgroups to identify performance disparities — consistent with ChiEAC's 
   fairness evaluation methodology.
5. **Deployment** — Deployed as a free Streamlit app accessible to non-technical 
   nonprofit staff.

---

## Future Improvements

- Add exact census tract boundaries using GeoJSON for precise map rendering
- Incorporate 2020 Census data for updated population figures
- Add time-series trend analysis using County Health Rankings historical data
- Build a nonprofit intake form to collect service area data directly

---

*Built as a portfolio project at ChiEAC | March 2026*
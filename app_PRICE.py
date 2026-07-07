import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Bike Rental Prediction Engine",
    page_icon="🚲",
    layout="centered"
)

st.markdown("""
    <style>
    .metric-container {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #E2E8F0;
        text-align: center;
        margin-top: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- These string labels match EXACTLY what was used in the training notebook
# --- (df["season"].map(...), df["mnth"].map(...), etc.) before pd.get_dummies().
SEASON_MAP = {1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'}
MONTH_MAP = {1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun',
             7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'}
WEEKDAY_MAP = {0: 'sun', 1: 'mon', 2: 'tue', 3: 'wed', 4: 'thu', 5: 'fri', 6: 'sat'}
WEATHER_MAP = {1: 'clear', 2: 'misty', 3: 'light snow/rain'}

# Friendly display labels for the UI (what the user sees in the dropdown)
DISPLAY = {
    'season': {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'},
    'yr': {0: '2018', 1: '2019'},
    'mnth': {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
             7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'},
    'weekday': {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'},
    'weathersit': {1: 'Clear / Partly Cloudy', 2: 'Mist / Cloudy', 3: 'Light Snow / Rain'}
}
REV_DISPLAY = {col: {v: k for k, v in d.items()} for col, d in DISPLAY.items()}


@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("bike_rental_model.pkl")
        scaler = joblib.load("price_scaler.pkl")
        return model, scaler
    except FileNotFoundError as e:
        st.error(
            f"❌ Missing artifact: {e.filename}. Please ensure both "
            f"'bike_rental_model.pkl' and 'price_scaler.pkl' are in the same "
            f"folder as this script."
        )
        st.stop()


model, scaler = load_artifacts()

# The GridSearchCV object stores the training-time column names/order here.
# Falls back to best_estimator_ just in case.
try:
    MODEL_COLUMNS = list(model.feature_names_in_)
except AttributeError:
    MODEL_COLUMNS = list(model.best_estimator_.feature_names_in_)

st.title("🚲 Bike Rental Prediction Engine")
st.caption("Adjust the parameters below to see the model's live demand prediction.")

col1, col2 = st.columns(2)

with col1:
    ui_season = st.selectbox("Season", list(DISPLAY['season'].values()), index=2)
    ui_yr = st.selectbox("Year Context", list(DISPLAY['yr'].values()), index=1)
    ui_mnth = st.selectbox("Month", list(DISPLAY['mnth'].values()), index=8)
    ui_weekday = st.selectbox("Day of Week", list(DISPLAY['weekday'].values()), index=2)
    ui_weathersit = st.selectbox("Weather Condition", list(DISPLAY['weathersit'].values()), index=0)

with col2:
    ui_holiday = st.selectbox("Is it a Holiday?", ["No", "Yes"])
    ui_workingday = st.selectbox("Is it a Working Day?", ["Yes", "No"])
    ui_temp = st.slider("Temperature (temp)", 0.0, 1.0, 0.5)
    ui_atemp = st.slider("Feeling Temp (atemp)", 0.0, 1.0, 0.5)
    ui_hum = st.slider("Humidity (hum)", 0.0, 1.0, 0.6)
    ui_windspeed = st.slider("Windspeed (windspeed)", 0.0, 1.0, 0.2)

# Map UI selections back to the raw integer codes, then to the exact
# string labels used at training time.
season_code = REV_DISPLAY['season'][ui_season]
yr_code = REV_DISPLAY['yr'][ui_yr]
mnth_code = REV_DISPLAY['mnth'][ui_mnth]
weekday_code = REV_DISPLAY['weekday'][ui_weekday]
weathersit_code = REV_DISPLAY['weathersit'][ui_weathersit]

raw_input = pd.DataFrame([{
    'season': SEASON_MAP[season_code],
    'yr': yr_code,
    'mnth': MONTH_MAP[mnth_code],
    'holiday': 1 if ui_holiday == "Yes" else 0,
    'weekday': WEEKDAY_MAP[weekday_code],
    'workingday': 1 if ui_workingday == "Yes" else 0,
    'weathersit': WEATHER_MAP[weathersit_code],
    'temp': ui_temp,
    'atemp': ui_atemp,
    'hum': ui_hum,
    'windspeed': ui_windspeed
}])

# 1. Scale the numeric features exactly as done in training.
scaled_features = ['temp', 'atemp', 'hum', 'windspeed']
raw_input[scaled_features] = scaler.transform(raw_input[scaled_features])

# 2. One-hot encode the categoricals the same way (drop_first=True in training
#    means the "baseline" category for each column simply produces no dummy
#    column here — reindexing below fills those correctly with 0).
processed_input = pd.get_dummies(
    raw_input,
    columns=['season', 'mnth', 'weekday', 'weathersit'],
    dtype=int
)

# 3. Align to the exact columns/order the model was trained on.
processed_input = processed_input.reindex(columns=MODEL_COLUMNS, fill_value=0)

prediction = model.predict(processed_input)[0]

if ui_holiday == "Yes" and ui_workingday == "Yes":
    st.warning(
        "⚠️ You've selected both 'Holiday' and 'Working Day' — this combination "
        "rarely appears in the training data, so the prediction may be unreliable."
    )

st.markdown(f"""
    <div class="metric-container">
        <p style="margin:0; font-size:0.9rem; text-transform:uppercase; font-weight:600; letter-spacing:0.05em; color:#64748B;">
            Predicted Rental Demand (cnt)
        </p>
        <h1 style="margin:0.5rem 0 0 0; font-size:3.5rem; color:#2563EB; font-weight:800;">
            {int(max(0, prediction)):,}
        </h1>
    </div>
""", unsafe_allow_html=True)
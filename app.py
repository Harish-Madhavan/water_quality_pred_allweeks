# Import all the necessary libraries
import pandas as pd
import joblib
import streamlit as st

# --- Page Configuration (Set ONCE at the top) ---
st.set_page_config(page_title="Water Pollutants Predictor", page_icon="💧", layout="wide")

# --- Caching and Loading ---
@st.cache_resource
def load_model_and_columns():
    """Loads the ML model and column list with error handling."""
    try:
        model = joblib.load("pollution_model.pkl")
        model_cols = joblib.load("model_columns.pkl")
        return model, model_cols
    except FileNotFoundError:
        st.error("🚨 Model or column file not found. Make sure 'pollution_model.pkl' and 'model_columns.pkl' are in the same directory as your script.")
        st.stop()

# Load the model and structure
model, model_cols = load_model_and_columns()

# --- Data Preparation for UI ---
try:
    known_station_ids = sorted([col.replace('id_', '') for col in model_cols if col.startswith('id_')])
except Exception as e:
    st.error(f"Could not parse station IDs from model columns. Error: {e}")
    known_station_ids = []

# --- User Interface ---

# 1. Title and Introduction
st.title("💧 Water Pollutants Predictor 🧪")
st.markdown("Welcome! This app predicts the concentration of various pollutants in water based on the year and monitoring station ID.")
st.markdown("---")

# 2. Sidebar for User Inputs
st.sidebar.header("📝 Input Parameters")
st.sidebar.write("Select the year and station to get a prediction.")

year_input = st.sidebar.number_input("Enter Year", min_value=2000, max_value=2100, value=2023, step=1)

if known_station_ids:
    station_id = st.sidebar.selectbox("Select Station ID", options=known_station_ids)
else:
    st.sidebar.warning("Could not determine station IDs. Please enter one manually.")
    station_id = st.sidebar.text_input("Enter Station ID", value='1')

# 3. Prediction Button
predict_button = st.sidebar.button('🚀 Predict Pollutant Levels', type="primary")

# --- Prediction and Display Logic ---

if predict_button:
    # Prepare input DataFrame
    input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
    input_encoded = pd.get_dummies(input_df, columns=['id'])
    
    # Align columns
    final_input = pd.DataFrame(columns=model_cols).fillna(0)
    final_input = pd.concat([final_input, input_encoded])
    final_input.fillna(0, inplace=True)
    final_input = final_input[model_cols]

    # Predict
    predicted_pollutants = model.predict(final_input)[0]
    pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

    # Create results DataFrame
    results_df = pd.DataFrame({
        "Pollutant": pollutants,
        "Predicted Value (mg/L)": predicted_pollutants
    })

    st.header(f"📊 Prediction Results for Station '{station_id}' in {year_input}")

    # Layout for results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Predicted Values")
        st.dataframe(results_df.style.background_gradient(
            cmap='Reds', subset=['Predicted Value (mg/L)']
        ).format({"Predicted Value (mg/L)": "{:.3f}"}),
        use_container_width=True)

    with col2:
        st.subheader("📊 Pollutant Comparison")
        chart_data = results_df.set_index('Pollutant')
        st.bar_chart(chart_data)

    st.success("Prediction complete! The table and chart are displayed above.")
    st.markdown("---") # Visual separator

    # --- NEW: Add Download Button ---
    
    # Convert DataFrame to CSV format for download.
    # index=False prevents pandas from writing the DataFrame index as a column.
    # .encode('utf-8') converts the string to bytes, which the download button requires.
    csv = results_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"prediction_station_{station_id}_year_{year_input}.csv",
        mime='text/csv',
    )
    
else:
    # Show initial welcome message
    st.info("⬅️ Please enter your parameters in the sidebar to get a prediction.")
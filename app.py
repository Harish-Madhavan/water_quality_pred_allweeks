# Import all the necessary libraries
import pandas as pd
import joblib
import streamlit as st

# --- Page Configuration (Set ONCE at the top) ---
# Use st.set_page_config for a nicer app title and icon in the browser tab.
st.set_page_config(page_title="Water Pollutants Predictor", page_icon="💧", layout="wide")

# --- Caching and Loading ---
# Use st.cache_resource to load the model and columns only once.
@st.cache_resource
def load_model_and_columns():
    """Loads the ML model and column list with error handling."""
    try:
        model = joblib.load("pollution_model.pkl")
        model_cols = joblib.load("model_columns.pkl")
        return model, model_cols
    except FileNotFoundError:
        st.error("🚨 Model or column file not found. Make sure 'pollution_model.pkl' and 'model_columns.pkl' are in the same directory as your script.")
        st.stop() # Stop the app if files are missing

# Load the model and structure
model, model_cols = load_model_and_columns()

# --- Data Preparation for UI ---
# Dynamically create a list of known station IDs from the model columns.
try:
    known_station_ids = sorted([col.replace('id_', '') for col in model_cols if col.startswith('id_')])
except Exception as e:
    st.error(f"Could not parse station IDs from model columns. Error: {e}")
    known_station_ids = [] # Fallback to an empty list

# --- User Interface ---

# 1. Title and Introduction on the main page
st.title("💧 Water Pollutants Predictor 🧪")
st.markdown("Welcome! This app predicts the concentration of various pollutants in water based on the year and monitoring station ID.")
st.markdown("---")

# 2. Sidebar for User Inputs
st.sidebar.header("📝 Input Parameters")
st.sidebar.write("Select the year and station to get a prediction.")

year_input = st.sidebar.number_input("Enter Year", min_value=2000, max_value=2100, value=2023, step=1)

# Use a dropdown for station IDs for better UX and to prevent errors.
if known_station_ids:
    station_id = st.sidebar.selectbox("Select Station ID", options=known_station_ids)
else:
    # Fallback to text input if station IDs couldn't be parsed
    st.sidebar.warning("Could not determine station IDs. Please enter one manually.")
    station_id = st.sidebar.text_input("Enter Station ID", value='1')

# 3. Prediction Button
predict_button = st.sidebar.button('🚀 Predict Pollutant Levels', type="primary")

# --- Prediction and Display Logic ---

# Display results only when the button is clicked
if predict_button:
    # Prepare the input DataFrame from user selections
    input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})

    # One-hot encode the 'id' column
    input_encoded = pd.get_dummies(input_df, columns=['id'])

    # Align columns with the model's training columns
    final_input = pd.DataFrame(columns=model_cols).fillna(0)
    final_input = pd.concat([final_input, input_encoded])
    final_input.fillna(0, inplace=True)
    final_input = final_input[model_cols]

    # Predict using the prepared data
    predicted_pollutants = model.predict(final_input)[0]
    pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

    # Create a results DataFrame
    results_df = pd.DataFrame({
        "Pollutant": pollutants,
        "Predicted Value (mg/L)": predicted_pollutants
    })

    # Display the results in a more engaging way
    st.header(f"📊 Prediction Results for Station '{station_id}' in {year_input}")

    # Use columns for a side-by-side layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Predicted Values")
        # Style the table with a color gradient to highlight high values
        st.dataframe(results_df.style.background_gradient(
            cmap='Reds', subset=['Predicted Value (mg/L)']
        ).format({"Predicted Value (mg/L)": "{:.3f}"}),
        use_container_width=True)

    with col2:
        st.subheader("📊 Pollutant Comparison")
        # Create a bar chart for quick visual comparison
        chart_data = results_df.set_index('Pollutant')
        st.bar_chart(chart_data)

    st.success("Prediction complete! The table shows the precise values, and the chart provides a quick visual comparison.")

else:
    # Show an initial welcome message on the main page
    st.info("⬅️ Please enter your parameters in the sidebar to get a prediction.")
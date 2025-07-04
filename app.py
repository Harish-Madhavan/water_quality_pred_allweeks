# Import all the necessary libraries
import pandas as pd
import joblib
import streamlit as st

# --- Caching and Loading ---
# Use st.cache_resource to load the model and columns only once.
# This prevents reloading from disk on every user interaction, making the app much faster.
@st.cache_resource
def load_model_and_columns():
    """Loads the ML model and column list with error handling."""
    try:
        model = joblib.load("pollution_model.pkl")
        model_cols = joblib.load("model_columns.pkl")
        return model, model_cols
    except FileNotFoundError:
        st.error("Model or column file not found. Make sure 'pollution_model.pkl' and 'model_columns.pkl' are in the same directory as your script.")
        st.stop() # Stop the app if files are missing

# Load the model and structure
model, model_cols = load_model_and_columns()

# --- Data Preparation for UI ---
# Dynamically create a list of known station IDs from the model columns.
# This is much safer than free text input. It assumes columns are named 'id_STATIONID'.
try:
    known_station_ids = [col.replace('id_', '') for col in model_cols if col.startswith('id_')]
except Exception as e:
    st.error(f"Could not parse station IDs from model columns. Error: {e}")
    known_station_ids = [] # Fallback to an empty list

# --- User Interface ---
st.title("Water Pollutants Predictor")
st.write("Predict the water pollutants based on Year and Station ID.")

# User inputs
year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2023, step=1)

# Use a dropdown for station IDs for better UX and to prevent errors.
if known_station_ids:
    station_id = st.selectbox("Select Station ID", options=known_station_ids)
else:
    # Fallback to text input if station IDs couldn't be parsed
    st.warning("Could not determine station IDs. Please enter one manually.")
    station_id = st.text_input("Enter Station ID", value='1')

# To encode and then predict
if st.button('Predict'):
    # Prepare the input DataFrame from user selections
    input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
    
    # One-hot encode the 'id' column
    input_encoded = pd.get_dummies(input_df, columns=['id'])

    # Align columns with the model's training columns
    # Create a new DataFrame with the model's columns and fill with 0
    final_input = pd.DataFrame(columns=model_cols).fillna(0)
    # Combine with the user's input. Existing columns are filled, new ones are ignored.
    final_input = pd.concat([final_input, input_encoded])
    # Fill any remaining NaN values with 0
    final_input.fillna(0, inplace=True)
    # Ensure the column order is exactly what the model expects
    final_input = final_input[model_cols]

    # Predict using the prepared data
    predicted_pollutants = model.predict(final_input)[0]
    pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

    # Display the results in a clean table
    st.subheader(f"Predicted Pollutant Levels for Station '{station_id}' in {year_input}:")
    
    results_df = pd.DataFrame({
        "Pollutant": pollutants,
        "Predicted Value (units)": predicted_pollutants
    })
    
    st.table(results_df.style.format({"Predicted Value (units)": "{:.2f}"}))
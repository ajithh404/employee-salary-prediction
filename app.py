import streamlit as st
import pandas as pd
import joblib
import os

try:
    model = joblib.load("best_salary_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    scaler = joblib.load("scaler.pkl")
    st.success("Model, encoders, and scaler loaded successfully!")
except FileNotFoundError:
    st.error("Error: One or more necessary files (best_salary_model.pkl, label_encoders.pkl, scaler.pkl) not found. Please ensure they are in the same directory as app.py and run the model training notebook first.")
    st.stop() 

# Load the original dataset for sample records 

try:
    original_df = pd.read_csv('salary_prediction_data.csv')
except FileNotFoundError:
    st.warning("Original dataset 'salary_prediction_data.csv' not found. Sample records cannot be displayed.")
    original_df = None 

st.set_page_config(page_title="Employee Salary Prediction", page_icon="💰", layout="centered")

st.title("💰 Employee Salary Prediction App ")
st.markdown("Predict an employee's salary based on their details.")


st.sidebar.header("✨ Tell Us About the Employee! ✨")
st.sidebar.markdown("Adjust the parameters below to see an estimated salary.")

# Define unique categories for select boxes based on the training data
education_options = list(encoders['Education'].classes_)
location_options = list(encoders['Location'].classes_)
job_title_options = list(encoders['Job_Title'].classes_)
gender_options = list(encoders['Gender'].classes_)

age = st.sidebar.slider(
    "🎂 Age",
    18, 75, 30,
    help="Enter the employee's current age. Age often correlates with experience and career stage."
)
experience = st.sidebar.slider(
    "⏳ Experience (Years)",
    0, 40, 5,
    help="How many years of professional experience does the employee have? More experience often leads to higher salaries."
)
education = st.sidebar.selectbox(
    "🎓 Education Level",
    education_options,
    help="Select the highest level of education attained by the employee. Higher education typically opens doors to better-paying roles."
)
location = st.sidebar.selectbox(
    "📍 Location",
    location_options,
    help="Choose the geographical location of the employee. Salaries can vary significantly based on cost of living and market demand in different regions."
)
job_title = st.sidebar.selectbox(
    "💼 Job Title",
    job_title_options,
    help="Select the employee's current job title. This is a strong indicator of responsibilities and industry standard pay."
)
gender = st.sidebar.selectbox(
    "🚻 Gender",
    gender_options,
    help="Specify the employee's gender. While salary should be equitable, gender can sometimes be a factor in historical data patterns."
)

# Preprocess input for single prediction 
def preprocess_input(age, experience, education, location, job_title, gender, encoders, scaler):
    input_data = pd.DataFrame({
        'Education': [education],
        'Experience': [experience],
        'Location': [location],
        'Job_Title': [job_title],
        'Age': [age],
        'Gender': [gender]
    })


    for col, encoder in encoders.items():
        if input_data[col][0] not in encoder.classes_:
            st.warning(f"Warning: '{input_data[col][0]}' is not a known category for '{col}'. Prediction might be inaccurate.")
            input_data[col] = -1
        else:
            input_data[col] = encoder.transform(input_data[col])

    input_data_ordered = input_data[['Education', 'Experience', 'Location', 'Job_Title', 'Age', 'Gender']]
    scaled_input = scaler.transform(input_data_ordered)
    return scaled_input

# Display Input Data 
st.write("### 🔎 Input Data for Single Prediction")
input_df_display = pd.DataFrame({
    'Age': [age],
    'Experience': [experience],
    'Education': [education],
    'Location': [location],
    'Job_Title': [job_title],
    'Gender': [gender]
})
st.write(input_df_display.to_markdown(index=False, numalign="left", stralign="left"))

# Predict button for single prediction 
if st.button("Predict Salary"):
    if age <= 0 or experience < 0:
        st.error("Age must be positive and Experience cannot be negative.")
    else:
        processed_input = preprocess_input(age, experience, education, location, job_title, gender, encoders, scaler)
        prediction = model.predict(processed_input)[0]
        st.success(f"💰 Predicted Salary: ${prediction:,.2f}")
        st.balloons()


try:
    model_performance_df = joblib.load("model_performance_metrics.pkl")
    st.success("Model performance metrics loaded successfully!")
except FileNotFoundError:
    st.warning("Model performance metrics file 'model_performance_metrics.pkl' not found. Model evaluation table cannot be displayed.")
    model_performance_df = None # Set to None if not found
st.set_page_config(page_title="Employee Salary Prediction", page_icon="💰", layout="centered")


# Model Performance Section 
st.markdown("---")
st.header("📈 Model Performance Overview")
if model_performance_df is not None:
    st.write("A summary comparing model performance on test data — higher R² indicates better fit, while lower MAE, RMSE, and MAPE reflect fewer prediction errors.")
    st.dataframe(model_performance_df.style.format({
        'Test R2': "{:.4f}",
        'Test MAE': "{:,.2f}",
        'Test RMSE': "{:,.2f}",
        'Test MAPE': "{:,.2f}%"
    }))
    # Displaying the best model based on R2 score
    st.info(f"The best performing model selected for this app is: **{model_performance_df['Test R2'].idxmax()}**")
else:
    st.info("Model performance metrics not available. Please ensure 'model_performance_metrics.pkl' is generated by running the model training notebook.")


# Sample Records Section
st.markdown("---")
st.header("📊 Sample Records from Dataset")
if original_df is not None:
    st.write("20 Sample records (random) from the original dataset:")
    st.write(original_df.sample(20, random_state=42).to_markdown(index=False, numalign="left", stralign="left"))
else:
    st.info("Cannot display sample records: 'salary_prediction_data.csv' was not found during app initialization.")


# Batch prediction section
st.markdown("---")
st.markdown("#### 📂 Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:")
    st.write(batch_data.head().to_markdown(index=False, numalign="left", stralign="left"))

    try:
        batch_data_processed = batch_data.copy()

        for col, encoder in encoders.items():
            if col in batch_data_processed.columns:
                batch_data_processed[col] = batch_data_processed[col].apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
            else:
                st.warning(f"Column '{col}' not found in uploaded batch data. Skipping encoding for this column.")

        expected_columns = ['Education', 'Experience', 'Location', 'Job_Title', 'Age', 'Gender']
        if not all(col in batch_data_processed.columns for col in expected_columns):
            st.error(f"Error: Batch file must contain all required columns: {', '.join(expected_columns)}")
        else:
            batch_data_for_scaling = batch_data_processed[expected_columns]
            scaled_batch_data = scaler.transform(batch_data_for_scaling)

            batch_preds = model.predict(scaled_batch_data)
            batch_data['Predicted_Salary'] = batch_preds

            st.write("✅ Predictions:")
            st.write(batch_data.head().to_markdown(index=False, numalign="left", stralign="left"))

            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Predictions CSV",
                csv,
                file_name='predicted_salaries.csv',
                mime='text/csv'
            )
    except Exception as e:
        st.error(f"An error occurred during batch prediction: {e}")

# Footer
st.markdown("---", unsafe_allow_html=True)

st.markdown(
    """
    <div style="
        border: 1px solid rgba(120, 120, 120, 0.3);
        padding: 12px;
        border-radius: 8px;
        background-color: rgba(240, 240, 240, 0.1);
        text-align: center;
        font-size: 14px;
        color: inherit;
    ">
        Project made as part of <strong>EDUNET SKILLSBUILD IBM AI Internship</strong> by <strong>Ajith Kumar</strong>
    </div>
    """,
    unsafe_allow_html=True
)

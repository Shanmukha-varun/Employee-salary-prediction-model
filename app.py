import streamlit as st
import pandas as pd
import joblib
# No need for StandardScaler or OneHotEncoder directly here, as they are part of the saved pipeline

# --- 1. Load the Trained Model ---
# Ensure your model file is in the same directory as this app.py
try:
    model_pipeline = joblib.load('best_model_pipeline_GradientBoosting.joblib')
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model file 'best_model_pipeline_GradientBoosting.joblib' not found. "
             "Please ensure it's in the same directory as app.py.")
    st.stop() # Stop the app if model is not found

# --- 2. Define Expected Feature Categories ---
# These must exactly match the categories used during training for OneHotEncoder (inside the pipeline implicitly)
# You would ideally extract these from your training data or a metadata file.
# For now, let's hardcode based on common adult.csv categories.
# Double-check these against your actual unique values in the original 'data' DataFrame
# after '?' handling and before one-hot encoding.

workclass_options = ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked']
marital_status_options = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
occupation_options = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces']
relationship_options = ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']
race_options = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
gender_options = ['Male', 'Female']
native_country_options = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Columbia', 'Poland', 'Japan', 'Iran', 'Taiwan', 'Haiti', 'Portugal', 'Nicaragua', 'Peru', 'France', 'Ecuador', 'Greece', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Cambodia', 'Laos', 'Thailand', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Honduras', 'Scotland', 'Holand-Netherlands']


# --- 3. Streamlit App Interface ---
st.set_page_config(page_title="Salary Prediction App", layout="centered")

st.title("💰 Employee Income Classifier")
st.write("Enter the employee's details to predict if their income is >50K or <=50K.")

# Input fields for numerical features
st.subheader("Personal Information")
age = st.slider("Age", 17, 90, 30) # Min, Max, Default
educational_num = st.slider("Educational Number (e.g., 9 for HS-grad, 13 for Bachelors)", 1, 16, 9)
hours_per_week = st.slider("Hours per Week", 1, 99, 40)
capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, max_value=100000, value=0)

# Input fields for categorical features (using selectbox)
st.subheader("Work & Marital Status")
workclass = st.selectbox("Workclass", options=workclass_options)
marital_status = st.selectbox("Marital Status", options=marital_status_options)
occupation = st.selectbox("Occupation", options=occupation_options)
relationship = st.selectbox("Relationship", options=relationship_options)

st.subheader("Demographics")
race = st.selectbox("Race", options=race_options)
gender = st.selectbox("Gender", options=gender_options)
native_country = st.selectbox("Native Country", options=native_country_options)


# --- 4. Create DataFrame from User Input ---
# This DataFrame's column names and order MUST match your training data *before* OneHotEncoding
# but *after* initial preprocessing (like 'education' removal if applicable)
# and before scaling.

# Note: 'fnlwgt' was likely a feature too. We need to handle if you excluded it or what its typical value is.
# Assuming fnlwgt was used, a typical median value can be used as a placeholder if not collected from user.
# For simplicity, let's include it assuming it was used during training and give a median default.
# If you dropped fnlwgt earlier, remove it from this dict.
fnlwgt_placeholder = 178300 # A median value from typical adult.csv datasets

input_data = pd.DataFrame({
    'age': [age],
    'fnlwgt': [fnlwgt_placeholder], # Placeholder for fnlwgt
    'educational-num': [educational_num],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'workclass': [workclass],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'native-country': [native_country]
})

# --- Important: Ensure column order matches X_train BEFORE scaling/encoding ---
# Get the exact column order from your X_train before it was scaled/encoded in the pipeline.
# If your X_train was already scaled/encoded before the pipeline, this part needs adjustment.
# But assuming X_train to the pipeline was original features, this is correct.
# You might need to adjust this list based on your X.columns before the final Pipeline was built.
original_feature_order = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss',
                         'hours-per-week', 'workclass', 'marital-status', 'occupation',
                         'relationship', 'race', 'gender', 'native-country']

# Ensure the input_data DataFrame has the same columns and order as the original features
input_data = input_data[original_feature_order]


# --- 5. Make Prediction ---
if st.button("Predict Income"):
    # The loaded pipeline handles both OneHotEncoding (implicitly if done on original X)
    # and StandardScaler internally.
    prediction = model_pipeline.predict(input_data)

    st.subheader("Prediction Result:")
    if prediction[0] == True: # Assuming True maps to '>50K' based on your y target
        st.success("The model predicts this individual's income is **>50K** 🎉")
    else:
        st.info("The model predicts this individual's income is **<=50K** 😔")

    st.write("---")
    st.write("Disclaimer: This is a machine learning prediction based on the provided inputs and historical data. It should not be used as financial advice.")

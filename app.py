import streamlit as st
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- 1. Load the Trained Model ---
# Ensure your model file is in the same directory as this app.py
try:
    # Your saved model is a pipeline: ('scaler', StandardScaler()), ('model', GradientBoostingClassifier())
    # This means X_train was already one-hot encoded before being fed into this pipeline during training.
    model_pipeline = joblib.load('best_model_pipeline_GradientBoosting.joblib')
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model file 'best_model_pipeline_GradientBoosting.joblib' not found. "
             "Please ensure it's in the same directory as app.py.")
    st.stop() # Stop the app if model is not found

# --- 2. Define Expected Feature Categories and Feature Names ---
# THESE MUST EXACTLY MATCH WHAT WAS USED DURING TRAINING.
# If these lists are not perfect, the model prediction will be incorrect or fail.

# Numerical columns - from your data_scaled, these are the original numerical features
numerical_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# Categorical columns - these are the original categorical features that need one-hot encoding
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']

# Options for Streamlit selectboxes - These must contain ALL categories present in your training data
workclass_options = ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked']
marital_status_options = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
occupation_options = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces']
relationship_options = ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']
race_options = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
gender_options = ['Male', 'Female']
native_country_options = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Columbia', 'Poland', 'Japan', 'Iran', 'Taiwan', 'Haiti', 'Portugal', 'Nicaragua', 'Peru', 'France', 'Ecuador', 'Greece', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Cambodia', 'Laos', 'Thailand', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Honduras', 'Scotland', 'Holand-Netherlands']


# --- 3. Define the Preprocessor for One-Hot Encoding ---
# This ColumnTransformer will apply OneHotEncoder to categorical columns.
# remainder='passthrough' ensures numerical columns are kept.
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough' # Keep numerical columns
)

# To ensure the preprocessor learns the correct column order and names for OHE,
# we need to fit it on a dummy DataFrame that mimics your training data's structure.
# This ensures consistency in the output feature names and order.
sample_data_for_preprocessor_fit = pd.DataFrame({
    'age': [30], 'fnlwgt': [178300], 'educational-num': [9], 'capital-gain': [0],
    'capital-loss': [0], 'hours-per-week': [40], # Make sure 'hours-per-week' is spelled correctly (hyphen)
    'workclass': ['Private'], 'marital-status': ['Never-married'], 'occupation': ['Prof-specialty'],
    'relationship': ['Not-in-family'], 'race': ['White'], 'gender': ['Male'],
    'native-country': ['United-States']
})
# Ensure the dummy data's columns are in the correct order for the preprocessor
sample_data_for_preprocessor_fit = sample_data_for_preprocessor_fit[numerical_cols + categorical_cols]

# Fit the preprocessor on the dummy data to learn categories and output feature names
preprocessor.fit(sample_data_for_preprocessor_fit)

# Get the names of the features after preprocessing (numerical + one-hot encoded categorical)
# This order is based on how ColumnTransformer handles 'remainder' and 'transformers'.
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
final_feature_names = list(cat_feature_names) + numerical_cols # OneHotEncoder features come first, then remainder

# --- 4. Streamlit App Interface ---
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
# Set default values for selectboxes using index to prevent initial errors
workclass = st.selectbox("Workclass", options=workclass_options, index=workclass_options.index('Private'))
marital_status = st.selectbox("Marital Status", options=marital_status_options, index=marital_status_options.index('Never-married'))
occupation = st.selectbox("Occupation", options=occupation_options, index=occupation_options.index('Prof-specialty'))
relationship = st.selectbox("Relationship", options=relationship_options, index=relationship_options.index('Not-in-family'))

st.subheader("Demographics")
race = st.selectbox("Race", options=race_options, index=race_options.index('White'))
gender = st.selectbox("Gender", options=gender_options, index=gender_options.index('Male'))
native_country = st.selectbox("Native Country", options=native_country_options, index=native_country_options.index('United-States'))


# --- 5. Create DataFrame from User Input ---
# This DataFrame's column names and order MUST match your original X DataFrame
# BEFORE any one-hot encoding or scaling.
fnlwgt_placeholder = 178300 # A median value from typical adult.csv datasets, used as a constant

input_data_raw = pd.DataFrame({
    'age': [age],
    'fnlwgt': [fnlwgt_placeholder],
    'educational-num': [educational_num],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week], # Ensure correct column name 'hours-per-week' (hyphen)
    'workclass': [workclass],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'native-country': [native_country]
})

# Reorder columns of the raw input to match what the 'preprocessor' expects (numerical + categorical)
input_data_raw = input_data_raw[numerical_cols + categorical_cols]


# --- 6. Make Prediction ---
if st.button("Predict Income"):
    # Apply the one-hot encoding using the fitted preprocessor
    transformed_input_array = preprocessor.transform(input_data_raw)

    # Convert the transformed NumPy array back to a DataFrame with correct column names
    # This DataFrame is now ready for the StandardScaler inside your model_pipeline.
    processed_input_df = pd.DataFrame(transformed_input_array, columns=final_feature_names)

    # Make prediction using the loaded pipeline (which handles StandardScaler and then the model)
    prediction = model_pipeline.predict(processed_input_df)

    st.subheader("Prediction Result:")
    if prediction[0] == True: # Assuming True maps to '>50K' based on your y target
        st.success("The model predicts this individual's income is **>50K** 🎉")
    else:
        st.info("The model predicts this individual's income is **<=50K** 😔")

    st.write("---")
    st.write("Disclaimer: This is a machine learning prediction based on the provided inputs and historical data. It should not be used as financial advice.")

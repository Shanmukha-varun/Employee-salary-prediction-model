import streamlit as st
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer # New import
from sklearn.preprocessing import OneHotEncoder, StandardScaler # New imports, if not already in your pipeline's first step

# --- 1. Load the Trained Model ---
try:
    # Your saved model is a pipeline: ('scaler', StandardScaler()), ('model', GradientBoostingClassifier())
    model_pipeline = joblib.load('best_model_pipeline_GradientBoosting.joblib')
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model file 'best_model_pipeline_GradientBoosting.joblib' not found. "
             "Please ensure it's in the same directory as app.py.")
    st.stop()

# --- 2. Define Expected Feature Categories and Feature Names ---
# THESE MUST EXACTLY MATCH WHAT WAS USED DURING TRAINING, ESPECIALLY FOR ONE-HOT ENCODING
# This is crucial for your ColumnTransformer below.
# You might need to retrieve these from your original training script or a metadata file.

# Numerical columns
numerical_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# Categorical columns and their unique options as observed in training data
# Double-check these against your actual unique values in the original 'data' DataFrame
# after '?' handling and before one-hot encoding.
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']

workclass_options = ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked']
marital_status_options = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
occupation_options = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces']
relationship_options = ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']
race_options = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
gender_options = ['Male', 'Female']
native_country_options = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Columbia', 'Poland', 'Japan', 'Iran', 'Taiwan', 'Haiti', 'Portugal', 'Nicaragua', 'Peru', 'France', 'Ecuador', 'Greece', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Cambodia', 'Laos', 'Thailand', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Honduras', 'Scotland', 'Holand-Netherlands']

# --- Important: Reconstruct the ColumnTransformer used during training ---
# This is critical. You need to apply the SAME one-hot encoding logic as you did on your X_train.
# If your X was already one-hot encoded when you created the pipeline:
# THEN YOUR PIPELINE ONLY HAS StandardScaler, AND YOU NEED TO MANUALLY ONE-HOT ENCODE HERE.
#
# If your pipeline was: [('preprocessor', ColumnTransformer(...)), ('scaler', StandardScaler()), ('model', Model)]
# THEN the ColumnTransformer would be part of your saved model_pipeline, and you'd just pass raw data.
#
# Based on your previous code:
#   pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
# This implies StandardScaler was the first step, so X_train was ALREADY one-hot encoded.
# Therefore, we need to manually one-hot encode the input_data BEFORE passing it to the pipeline.

# Create a preprocessor for one-hot encoding the categorical features
# This preprocessor must be FIT on your X_train's categorical columns to learn the categories.
# Since we don't have the original X_train here, we'll initialize a new one.
# It's better practice to save and load this preprocessor too, if it was separate from the model pipeline.
# For now, we simulate the OneHotEncoder's behavior by explicitly providing known categories.

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough' # Keep numerical columns
)

# You need to fit this preprocessor on some data to get the feature names
# A simple way to do this here is to fit it on a dummy DataFrame with all expected columns
# or ideally, save the fitted preprocessor from your training script.
# For simplicity in the app, we'll try to get feature names after fitting.

# --- 3. Streamlit App Interface ---
st.set_page_config(page_title="Salary Prediction App", layout="centered")

st.title("💰 Employee Income Classifier")
st.write("Enter the employee's details to predict if their income is >50K or <=50K.")

# Input fields for numerical features
st.subheader("Personal Information")
age = st.slider("Age", 17, 90, 30)
educational_num = st.slider("Educational Number (e.g., 9 for HS-grad, 13 for Bachelors)", 1, 16, 9)
hours_per_week = st.slider("Hours per Week", 1, 99, 40)
capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, max_value=100000, value=0)

# Input fields for categorical features (using selectbox)
st.subheader("Work & Marital Status")
workclass = st.selectbox("Workclass", options=workclass_options, index=workclass_options.index('Private')) # Set default
marital_status = st.selectbox("Marital Status", options=marital_status_options, index=marital_status_options.index('Never-married'))
occupation = st.selectbox("Occupation", options=occupation_options, index=occupation_options.index('Prof-specialty'))
relationship = st.selectbox("Relationship", options=relationship_options, index=relationship_options.index('Not-in-family'))

st.subheader("Demographics")
race = st.selectbox("Race", options=race_options, index=race_options.index('White'))
gender = st.selectbox("Gender", options=gender_options, index=gender_options.index('Male'))
native_country = st.selectbox("Native Country", options=native_country_options, index=native_country_options.index('United-States'))

# --- 4. Create DataFrame from User Input ---
# This DataFrame should contain columns in the order expected by the ColumnTransformer's remainder='passthrough'
# and then by the one-hot encoding output.
# Need to use a placeholder for 'fnlwgt' as it's not taken from user input directly.
fnlwgt_placeholder = 178300 # A median value from typical adult.csv datasets

input_data_raw = pd.DataFrame({
    'age': [age],
    'fnlwgt': [fnlwgt_placeholder],
    'educational-num': [educational_num],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per_week': [hours_per_week], # Ensure correct column name 'hours-per-week'
    'workclass': [workclass],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'native-country': [native_country]
})

# Reorder columns to match the order expected by the preprocessor (numerical first, then categorical as defined)
input_data_raw = input_data_raw[numerical_cols + categorical_cols]

# --- 5. Fit a dummy preprocessor to get feature names ---
# This is a workaround. The best practice is to save your *fitted* preprocessor from training
# or to ensure your model_pipeline itself handles all preprocessing (OneHotEncoder + StandardScaler).
# If your training pipeline was Pipeline([('preprocessor', ColumnTransformer(...)), ('scaler', StandardScaler()), ('model', Model)]),
# then you wouldn't need this manual step.
#
# Here, we'll create a dummy dataset that reflects the structure of your X_train
# before one-hot encoding, just to fit the preprocessor and get column names.
# This assumes that the *order* of numerical and categorical columns in the original X_train
# before one-hot encoding was (numerical_cols + categorical_cols).

dummy_X_train = pd.DataFrame(columns=numerical_cols + categorical_cols)
# Add dummy rows if preprocessor complains about empty fit
# For now, we assume fitting on empty columns is okay for getting transformer names.

# Fit the preprocessor to a dummy DataFrame to learn the categorical features and their order
# This ensures it generates the same columns as during training.
# Using a small, representative DataFrame based on your options:
sample_data_for_preprocessor_fit = pd.DataFrame({
    'age': [30], 'fnlwgt': [178300], 'educational-num': [9], 'capital-gain': [0],
    'capital-loss': [0], 'hours-per_week': [40],
    'workclass': ['Private'], 'marital-status': ['Never-married'], 'occupation': ['Prof-specialty'],
    'relationship': ['Not-in-family'], 'race': ['White'], 'gender': ['Male'],
    'native-country': ['United-States']
})
# Ensure column order
sample_data_for_preprocessor_fit = sample_data_for_preprocessor_fit[numerical_cols + categorical_cols]

preprocessor.fit(sample_data_for_preprocessor_fit) # Fit the preprocessor

# --- 6. Transform User Input and Make Prediction ---
if st.button("Predict Income"):
    # Apply the one-hot encoding using the preprocessor
    transformed_input = preprocessor.transform(input_data_raw)

    # Convert to DataFrame to ensure column names are consistent if needed later
    # and to confirm shape.
    # Get feature names after one-hot encoding for the ColumnTransformer
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    final_features_names = numerical_cols + list(cat_feature_names) # This order depends on remainder='passthrough'
                                                                      # putting numericals first

    transformed_input_df = pd.DataFrame(transformed_input, columns=final_features_names)

    # The loaded model_pipeline itself starts with StandardScaler, which expects
    # a DataFrame of all numerical (or already scaled numericals) features.
    # So, we pass the fully transformed (one-hot encoded and numericals kept) data to the pipeline.
    prediction = model_pipeline.predict(transformed_input_df) # Pass the fully prepared data

    st.subheader("Prediction Result:")
    if prediction[0] == True: # Assuming True maps to '>50K' based on your y target
        st.success("The model predicts this individual's income is **>50K** 🎉")
    else:
        st.info("The model predicts this individual's income is **<=50K** 😔")

    st.write("---")
    st.write("Disclaimer: This is a machine learning prediction based on the provided inputs and historical data. It should not be used as financial advice.")

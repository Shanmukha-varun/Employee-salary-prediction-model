import streamlit as st
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- 0. Exact Model Input Columns from Training ---
# This list is CRUCIAL and must be EXACTLY as it was during your model training.
# You provided this list directly from your X_train.columns.tolist()
EXACT_MODEL_INPUT_COLUMNS = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'workclass_Local-gov', 'workclass_Others', 'workclass_Private', 'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc', 'workclass_State-gov', 'marital-status_Married-AF-spouse', 'marital-status_Married-civ-spouse', 'marital-status_Married-spouse-absent', 'marital-status_Never-married', 'marital-status_Separated', 'marital-status_Widowed', 'occupation_Armed-Forces', 'occupation_Craft-repair', 'occupation_Exec-managerial', 'occupation_Farming-fishing', 'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct', 'occupation_Other-service', 'occupation_Others', 'occupation_Priv-house-serv', 'occupation_Prof-specialty', 'occupation_Protective-serv', 'occupation_Sales', 'occupation_Tech-support', 'occupation_Transport-moving', 'relationship_Not-in-family', 'relationship_Other-relative', 'relationship_Own-child', 'relationship_Unmarried', 'relationship_Wife', 'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White', 'gender_Male', 'native-country_Cambodia', 'native-country_Canada', 'native-country_China', 'native-country_Columbia', 'native-country_Cuba', 'native-country_Dominican-Republic', 'native-country_Ecuador', 'native-country_El-Salvador', 'native-country_England', 'native-country_France', 'native-country_Germany', 'native-country_Greece', 'native-country_Guatemala', 'native-country_Haiti', 'native-country_Holand-Netherlands', 'native-country_Honduras', 'native-country_Hong', 'native-country_Hungary', 'native-country_India', 'native-country_Iran', 'native-country_Ireland', 'native-country_Italy', 'native-country_Jamaica', 'native-country_Japan', 'native-country_Laos', 'native-country_Mexico', 'native-country_Nicaragua', 'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_Peru', 'native-country_Philippines', 'native-country_Poland', 'native-country_Portugal', 'native-country_Puerto-Rico', 'native-country_Scotland', 'native-country_South', 'native-country_Taiwan', 'native-country_Thailand', 'native-country_Trinadad&Tobago', 'native-country_United-States', 'native-country_Vietnam', 'native-country_Yugoslavia']


# --- 1. Load the Trained Model ---
# Ensure your model file is in the same directory as this app.py
try:
    # Your saved model is a pipeline: ('scaler', StandardScaler()), ('model', GradientBoostingClassifier())
    # This implies that the data fed into this pipeline during training was ALREADY one-hot encoded.
    model_pipeline = joblib.load('best_model_pipeline_GradientBoosting.joblib')
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model file 'best_model_pipeline_GradientBoosting.joblib' not found. "
             "Please ensure it's in the same directory as app.py.")
    st.stop() # Stop the app if model is not found

# --- 2. Define Expected Feature Categories for Raw Input ---
# These are the *original* column names before any encoding.
numerical_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']

# Options for Streamlit selectboxes - These must contain ALL categories present in your training data
# from the 'raw' (unencoded) categorical columns.
workclass_options = ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked']
marital_status_options = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
occupation_options = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces']
relationship_options = ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']
race_options = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
gender_options = ['Male', 'Female']
native_country_options = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Columbia', 'Poland', 'Japan', 'Iran', 'Taiwan', 'Haiti', 'Portugal', 'Nicaragua', 'Peru', 'France', 'Ecuador', 'Greece', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Cambodia', 'Laos', 'Thailand', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Honduras', 'Scotland', 'Holand-Netherlands']


# --- 3. Define the Preprocessor for One-Hot Encoding Raw Input ---
# This ColumnTransformer will apply OneHotEncoder to categorical columns
# and pass through numerical columns.
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough' # Keep numerical columns (these will appear *after* OHE columns in output)
)

# To ensure the preprocessor learns the correct column names and order for OHE,
# we fit it on a dummy DataFrame that mimics your raw training data's structure.
sample_data_for_preprocessor_fit = pd.DataFrame({
    'age': [30], 'fnlwgt': [178300], 'educational-num': [9], 'capital-gain': [0],
    'capital-loss': [0], 'hours-per-week': [40],
    'workclass': ['Private'], 'marital-status': ['Never-married'], 'occupation': ['Prof-specialty'],
    'relationship': ['Not-in-family'], 'race': ['White'], 'gender': ['Male'],
    'native-country': ['United-States']
})
# Ensure the dummy data's columns are in the correct order for the preprocessor
sample_data_for_preprocessor_fit = sample_data_for_preprocessor_fit[numerical_cols + categorical_cols]
preprocessor.fit(sample_data_for_preprocessor_fit)


# --- 4. Streamlit App Interface ---
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
workclass = st.selectbox("Workclass", options=workclass_options, index=workclass_options.index('Private'))
marital_status = st.selectbox("Marital Status", options=marital_status_options, index=marital_status_options.index('Never-married'))
occupation = st.selectbox("Occupation", options=occupation_options, index=occupation_options.index('Prof-specialty'))
relationship = st.selectbox("Relationship", options=relationship_options, index=relationship_options.index('Not-in-family'))

st.subheader("Demographics")
race = st.selectbox("Race", options=race_options, index=race_options.index('White'))
gender = st.selectbox("Gender", options=gender_options, index=gender_options.index('Male'))
native_country = st.selectbox("Native Country", options=native_country_options, index=native_country_options.index('United-States'))


# --- 5. Create DataFrame from User Input (Raw Features) ---
fnlwgt_placeholder = 178300 # A median value, as fnlwgt is not a user input

input_data_raw = pd.DataFrame({
    'age': [age],
    'fnlwgt': [fnlwgt_placeholder],
    'educational-num': [educational_num],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week], # Ensure correct spelling with hyphen
    'workclass': [workclass],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'native-country': [native_country]
})

# Reorder columns of the raw input to match what the 'preprocessor' expects (numerical + categorical)
# This order is important for the ColumnTransformer to correctly identify numerical and categorical features.
input_data_raw = input_data_raw[numerical_cols + categorical_cols]


# --- 6. Transform User Input and Make Prediction ---
if st.button("Predict Income"):
    # Apply the one-hot encoding using the fitted preprocessor
    transformed_input_array = preprocessor.transform(input_data_raw)

    # Get the feature names as output by the preprocessor (OHE columns first, then numerical passthrough)
    preprocessor_output_feature_names = preprocessor.get_feature_names_out()

    # --- DEBUG LINE: Print the actual columns generated by the preprocessor ---
    st.write(f"**DEBUG: Columns generated by preprocessor:** {list(preprocessor_output_feature_names)}")
    st.write(f"**DEBUG: Expected model input columns:** {EXACT_MODEL_INPUT_COLUMNS}") # Also print expected for comparison

    # Create an unordered DataFrame from the preprocessor's output
    processed_input_df_unordered = pd.DataFrame(transformed_input_array, columns=preprocessor_output_feature_names)

    # Reorder the columns of this DataFrame to EXACTLY match the order your model was trained on.
    # This is the most critical step for the ValueError.
    processed_input_df = processed_input_df_unordered[EXACT_MODEL_INPUT_COLUMNS]

    st.subheader("Prediction Result:")
    if prediction[0] == True: # Assuming True maps to '>50K' based on your y target
        st.success("The model predicts this individual's income is **>50K** 🎉")
    else:
        st.info("The model predicts this individual's income is **<=50K** 😔")

    st.write("---")
    st.write("Disclaimer: This is a machine learning prediction based on the provided inputs and historical data. It should not be used as financial advice.")

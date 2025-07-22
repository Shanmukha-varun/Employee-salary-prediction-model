# Employee Salary Prediction Model

This repository hosts a machine learning project focused on predicting whether an individual's income is greater than \$50,000 or not, based on various demographic and employment-related features. The project includes data preprocessing, model training, evaluation, and a deployed Streamlit web application for interactive predictions.

## 🚀 Project Overview

The core problem addressed is the subjective and inefficient nature of manually estimating income brackets. Our solution leverages machine learning to provide an accurate, unbiased, and automated prediction system. This tool can be valuable for applications like targeted marketing, economic analysis, or resource allocation.

## ✨ Features

* **Data Preprocessing:** Handles missing values, performs feature scaling (Standardization), and categorical encoding (One-Hot Encoding).
* **Machine Learning Models:** Explores various classification algorithms including K-Nearest Neighbors, Logistic Regression, Random Forest, and Gradient Boosting.
* **Optimized Model:** Utilizes a **Gradient Boosting Classifier** for its high accuracy and robust performance.
* **Interactive Web Application:** A user-friendly interface built with Streamlit allows users to input data and receive real-time income predictions.
* **Model Persistence:** The trained model and preprocessor are saved to enable efficient deployment without retraining.

## 📁 Repository Structure

* `employee_salary_prediction_model.ipynb`: The main Jupyter Notebook containing all the steps from data acquisition, preprocessing, model training, and evaluation.
* `app.py`: The Python script for the Streamlit web application.
* `best_model_pipeline_GradientBoosting.joblib`: The saved trained machine learning model pipeline.
* `adult.csv`: The dataset used for this project.
* `README.md`: This file.

## 🛠️ Technologies & Libraries Used

* **Python 3.x**
* **`pandas`**: For data manipulation and analysis.
* **`numpy`**: For numerical operations.
* **`scikit-learn` (sklearn)**: The cornerstone ML library for:
    * `train_test_split`: Splitting data for training and testing.
    * `StandardScaler`: Scaling numerical features.
    * `OneHotEncoder`: Encoding categorical features.
    * `ColumnTransformer`: Applying diverse transformations to different columns.
    * `GradientBoostingClassifier`: The primary classification model used.
    * `accuracy_score`: Evaluating model performance.
* **`joblib`**: For saving and loading the trained model.
* **`streamlit`**: For building and deploying the interactive web application.

## 📊 Model Performance (Gradient Boosting Classifier)

* **Accuracy:** Approximately 84.11%.
* Further evaluation metrics like `classification_report` and `confusion_matrix` are available in the Jupyter Notebook.

## 🚀 How to Run the App Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/employee-salary-prediction-model.git](https://github.com/your-username/employee-salary-prediction-model.git)
    cd employee-salary-prediction-model
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You might need to create a `requirements.txt` file by running `pip freeze > requirements.txt` in your environment after installing all libraries.)*
4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The app will open in your web browser, typically at `http://localhost:8501`. 

## ☁️ Deployment

The application is deployed on Streamlit Cloud and can be accessed [Here](https://employee-salary-prediction-model.streamlit.app/) (Assuming this is your deployed URL).

## 📄 References

* **Jupyter Notebook:** `employee_salary_prediction_model.ipynb` (Primary source for code and analysis)
* **Dataset:** Adult Data Set (UCI Machine Learning Repository) - *[You might want to add the specific UCI URL here if you used it directly]*
* **Python Libraries Documentation:**
    * [Pandas Documentation](https://pandas.pydata.org/docs/)
    * [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
    * [Streamlit Documentation](https://docs.streamlit.io/)

---

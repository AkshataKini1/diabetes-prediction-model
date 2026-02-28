# Diabetes Prediction Model

A machine learning project to predict diabetes using the Pima Indians Diabetes Database, featuring Logistic Regression and Random Forest models.

### Project Description

This project aims to build and evaluate machine learning models to predict the onset of diabetes based on diagnostic measurements. The dataset used is the Pima Indians Diabetes Database, which contains medical predictor variables and one target variable, Outcome (0 for non-diabetic, 1 for diabetic).

The notebook covers the following key steps:

- Data Loading: Importing the dataset from KaggleHub.
- Exploratory Data Analysis (EDA): Initial exploration of the dataset, checking for missing values and data types.
- Data Preprocessing: Handling zero values in key features by replacing them with the mean, and splitting the data into training and testing sets.
- Model Training: Implementing and training two classification models:
  1. Logistic Regression
  2. Random Forest Classifier
- Model Evaluation: Assessing the performance of each model using metrics such as accuracy, confusion matrix, classification report, and ROC curves.
- Feature Importance: Analyzing feature importance for the Logistic Regression model.

## Dataset
- Source: [Pima Indians Diabetes Database on Kaggle](www.kaggle.com%2Fdatasets%2Fakshaydattatraykhare%2Fdiabetes-dataset)

## Features:
- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age (years)
- Outcome: Class variable (0 or 1) - 1 for diabetes, 0 for no diabetes

## Technologies Used
- Python 3
- pandas: For data manipulation and analysis.
- numpy: For numerical operations.
- scikit-learn: For machine learning models (Logistic Regression, RandomForestClassifier) and utilities (train_test_split, metrics, ColumnTransformer, SimpleImputer).
- seaborn: For statistical data visualization.
- matplotlib.pyplot: For plotting and visualization.
- plotly.express: For interactive visualizations (though not explicitly used in the final evaluation, it's imported in the setup).
- kagglehub: For easy dataset downloading.

## How to Run the Notebook
### Google Colab
1. Open in Google Colab: Click the "Open in Colab" badge or directly upload the .ipynb file to Google Colab.
2. Install Dependencies: Ensure all required libraries are installed (Colab typically has many pre-installed, but you can add !pip install <library_name> if needed).
3. Run Cells: Execute the cells sequentially to reproduce the analysis, model training, and evaluation.

### Jupyter Notebook or Anaconda
1. Clone the Repository: Clone this GitHub repository to your local machine.
2. Install Dependencies: Make sure you have Python 3 installed. You can install the required libraries using pip:
3. pip install pandas numpy scikit-learn seaborn matplotlib plotly kagglehub
4. Launch Jupyter: Open your terminal or Anaconda Prompt, navigate to the cloned repository directory, and run jupyter notebook.
5. Open and Run: Open the diabetes_prediction_model.ipynb (or whatever your notebook is named) file in your browser and execute the cells sequentially.

## Results Summary
The notebook compares the performance of Logistic Regression and Random Forest models for diabetes prediction. Both models achieve an accuracy of approximately 75-76% on the test set. Detailed precision, recall, and F1-score metrics are provided in the classification reports for both models.

# Customer Churn Prediction

## Project Overview

This project uses Machine Learning to predict whether a telecommunications customer is likely to leave a company (customer churn).

The objective is to identify customers at risk of leaving so businesses can take proactive retention actions.

The project uses the IBM Telco Customer Churn dataset and includes data exploration, preprocessing, model training, evaluation, and a Streamlit web application.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook
- Git and GitHub

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── raw/
│       └── telco_customer_churn.csv
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── models/
│   ├── decision_tree_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── results/
│
├── src/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Project Workflow

1. Data Collection and Understanding
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Feature Encoding and Scaling
5. Machine Learning Model Training
6. Model Evaluation
7. Model Saving
8. Streamlit Application Development

## Machine Learning Models

Three classification algorithms were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest

## Model Performance

The models were evaluated using Accuracy, Precision, Recall, and F1-score.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 73.81% | 50.43% | 78.34% | 61.36% |
| Decision Tree | 73.46% | 50.00% | 80.75% | 61.76% |
| Random Forest | 76.15% | 53.77% | 72.46% | 61.73% |

The Decision Tree was selected as the final model because it achieved the highest F1-score and recall for the churn class.

## Streamlit Web Application

The application allows users to:

- Enter customer information.
- Predict whether a customer is likely to churn.
- View the predicted churn probability.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/KATXENNN/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the environment

Mac/Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Launch the application

```bash
streamlit run app.py
```

## Evaluation

The final Decision Tree model achieved:

- Accuracy: 73.46%
- Churn Precision: 50%
- Churn Recall: 80.75%
- Churn F1-score: 61.76%

These results demonstrate the model's ability to identify customers at risk of churn.

## Future Improvements

- Hyperparameter tuning.
- Cross-validation.
- Model comparison and optimization.
- Improved probability calibration.
- Deployment to a cloud platform.

---

**Academic Project — Master 2**

*Customer Churn Prediction using Machine Learning and Python.*
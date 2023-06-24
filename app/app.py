import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder

# User defined functions
def transform_department(department):
    department_mapping = {
        'Sales': 'sales',
        'Technical': 'technical',
        'HR': 'hr',
        'Accounting': 'accounting',
        'Support': 'support',
        'Management': 'management',
        'IT': 'IT',
        'Product Management': 'product_mng',
        'Marketing': 'marketing',
        'RandD': 'RandD'
    }
    return department_mapping[department]

def transform_salary(salary_level):
    salary_mapping = {
        'Low': 'low',
        'Medium': 'medium',
        'High': 'high'
    }
    return salary_mapping[salary_level]

st.sidebar.title('Employee Churn Analysis')
html_temp = """
<div style="background-color:blue;padding:10px">
<h2 style="color:white;text-align:center;">Employee Churn Analysis App </h2>
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)

satisfaction_level = st.sidebar.slider("Employee Satisfaction Level", 0.0, 1.0, step=0.01)
last_evaluation = st.sidebar.slider("Last Evaluation Score", 0.0, 1.0, step=0.01)
number_projects = st.sidebar.slider("Number of Projects", 1, 10)
average_monthly_hours = st.sidebar.slider("Average Monthly Hours", 80, 400)
time_spent_company = st.sidebar.slider("Years at Company", 1, 10)
work_accident = st.sidebar.radio("Work Accident in the Past?", ('No', 'Yes'))
promotion_last_5years = st.sidebar.radio("Promotion in the Last 5 Years?", ('No', 'Yes'))
departments = st.sidebar.selectbox("Department", ('Sales', 'Technical', 'HR', 'Accounting', 'Support', 'Management', 'IT', 'Product Management', 'Marketing', 'RandD'))
salary = st.sidebar.selectbox("Salary Level", ('Low', 'Medium', 'High'))

employee_churn_model = pickle.load(open("gradient_boosting_model", "rb"))
with open('scaler.pkl', 'rb') as f:
    employee_churn_scaler = pickle.load(f)
with open('ordinal_encoder.pkl', 'rb') as f:
    employee_churn_encoder = pickle.load(f)

employee = {
    "satisfaction_level": satisfaction_level,
    "last_evaluation": last_evaluation,
    "number_projects": number_projects,
    "average_monthly_hours": average_monthly_hours,
    "time_spent_company": time_spent_company,
    "work_accident": work_accident,
    "promotion_last_5years": promotion_last_5years,
    "departments": departments,
    "salary": salary
}

df = pd.DataFrame.from_dict([employee])

st.header("The configuration of the employee is below")
st.table(df)

st.subheader("Press predict if the employee configuration is okay")

# Function to transform and prepare the data for prediction
def prepare_data(data):
    df_transformed = data.copy()
    df_transformed['departments'] = df_transformed.departments.apply(transform_department)
    df_transformed['salary'] = df_transformed.salary.apply(transform_salary)
    df_transformed[['departments', 'salary']] = employee_churn_encoder.transform(df_transformed[['departments', 'salary']])
    df_transformed['work_accident'] = df_transformed.work_accident.map({'No': 0, 'Yes': 1})
    df_transformed['promotion_last_5years'] = df_transformed.promotion_last_5years.map({'No': 0, 'Yes': 1})
    
    # Scaling the data 
    df_transformed = employee_churn_scaler.transform(df_transformed)
    return df_transformed

# Function to make prediction and display result
def predict_and_display(data):
    # Prepare data
    df_transformed = prepare_data(data)
    
    # Making prediction
    prediction = employee_churn_model.predict(df_transformed)
    prediction_proba = employee_churn_model.predict_proba(df_transformed)

    # Displaying prediction
    if prediction[0] == 0:
        st.success("The employee is predicted to stay in the company. 😊")
        st.text(f"Probability: {prediction_proba[0][0]*100:.2f}%")
        st.caption("The factors in favor of the employee's retention could be satisfactory salary, lower working accidents, recent promotions.")
    else:
        st.warning("The employee is predicted to leave the company. 😟")
        st.text(f"Probability: {prediction_proba[0][1]*100:.2f}%")
        st.caption("Potential reasons for the predicted departure might be dissatisfaction with salary, a high number of workplace accidents, lack of recent promotions.")

# Streamlit button for prediction
if st.button("Predict"):
    predict_and_display(df)

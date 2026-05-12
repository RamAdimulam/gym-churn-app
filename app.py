import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. App Header
st.title("Gym Member Churn Predictor")
st.write("Enter member details to predict the likelihood of cancellation.")

# 2. Input Fields (Based on your Feature Importance results)
st.sidebar.header("Member Data Input")
visits = st.sidebar.slider("Monthly Visits", 0, 30, 5)
spend = st.sidebar.number_input("Avg Monthly Spend ($)", 0, 500, 50)
age = st.sidebar.slider("Age", 18, 80, 30)
contract = st.sidebar.selectbox("Contract Period (Months)", [1, 6, 12])
group_class = st.sidebar.radio("Attends Group Classes?", [0, 1])

# 3. Model Logic
# Re-building the logic from your 1,000-record dataset
data = pd.read_csv('Gym_Churn_Final_Project_Data.csv')
X = data[['Monthly_Visits', 'Avg_Additional_Spend', 'Age', 'Contract_Period', 'Group_Class_Attendance']]
y = data['Churn']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Run Prediction
input_df = pd.DataFrame([[visits, spend, age, contract, group_class]], 
                         columns=['Monthly_Visits', 'Avg_Additional_Spend', 'Age', 'Contract_Period', 'Group_Class_Attendance'])

prediction = model.predict(input_df)
probability = model.predict_proba(input_df)[0][1]

# 5. Display Results
st.subheader("Churn Prediction Result")
if prediction[0] == 1:
    st.error(f"HIGH RISK: This member is {probability:.1%} likely to churn.")
    st.info("Recommendation: Targeted loyalty-based outreach required.")
else:
    st.success(f"LOW RISK: Member loyalty score is {(1-probability):.1%}.")
    st.info("Recommendation: Encourage group class participation to maintain habit.")

import streamlit as st
import torch
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import torch.nn as nn
# import torch.nn.functional as F
import torch.nn.functional as F

class Churn_Modelling(nn.Module):
    def __init__(self):
        super(Churn_Modelling, self).__init__()
        self.cf1 = nn.Linear(12, 64)  # input layer, 12 input features
        self.cf2 = nn.Linear(64, 32)  # hidden layer
        self.output = nn.Linear(32, 1)  # output layer

    def forward(self, x):
        x = F.relu(self.cf1(x))
        x = F.relu(self.cf2(x))
        x = torch.sigmoid(self.output(x))
        return x

# Load the model and encoders
model = torch.load('model.h5')
model.eval()

# Load the scaler and label encoder
with open('scaler.pkl', 'rb') as file:
    sc = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('one_hot_encoder.pkl', 'rb') as file:
    one_hot = pickle.load(file)

# Define the function for prediction
def predict(input_data):
    # Apply transformations to the input data
    input_df = pd.DataFrame(input_data, index=[0])
    # One-hot encode 'Geography' and 'Gender' columns
    encoded_geography = one_hot.transform(input_df[['Geography']]).toarray()
    input_df = input_df.drop('Geography', axis=1)
    encoded_geography_df = pd.DataFrame(encoded_geography, columns=one_hot.get_feature_names_out(['Geography']))
    input_df = pd.concat([input_df.reset_index(drop=True), encoded_geography_df], axis=1)

    input_df['Gender'] = label_encoder.transform(input_df['Gender'])

    expected_columns = [
        "CreditScore","Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", 
        "IsActiveMember", "EstimatedSalary", "Geography_France", "Geography_Germany", 
        "Geography_Spain" 
    ]
    input_df = input_df[expected_columns]

    # Scaling the input data
    scaled_input = sc.transform(input_df)

    # Convert the input data to a tensor
    inputs = torch.tensor(scaled_input, dtype=torch.float32)

    # Make the prediction
    with torch.no_grad():
        output = model(inputs)
        prediction = output.round().item()  # Convert tensor to scalar and round to 0 or 1

    return prediction

# Streamlit UI
st.title('Churn Prediction')

# Create input fields for the user
credit_score = st.slider('Credit Score', 300, 850, 600)
age = st.slider('Age', 18, 100, 30)
tenure = st.slider('Tenure', 0, 10, 5)
balance = st.number_input('Balance', value=10000.0)
num_of_products = st.slider('Number of Products', 1, 4, 2)
has_cr_card = st.selectbox('Has Credit Card?', [0,1])
is_active_member = st.selectbox('Is Active Member?', [0,1])
estimated_salary = st.number_input('Estimated Salary', value=50000.0)
geography = st.selectbox('Geography', one_hot.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)

# Prepare the input data as a dictionary
input_data = {
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': 1 if has_cr_card == 'Yes' else 0,
    'IsActiveMember': 1 if is_active_member == 'Yes' else 0,
    'EstimatedSalary': estimated_salary,
    'Geography': geography,
    'Gender': gender
}

# encoded_data = one_hot.transform([[geography]]).toarray()
# encoded_data_geography = pd.DataFrame(encoded_data,columns=one_hot.get_feature_names_out(['Geography']))

# input_data = pd.concat([input_data.reset_index(drop=True) ,encoded_data_geography],axis=1)

# input_data_scaled = sc.transform(input_data)

# Prediction button
if st.button('Predict Churn'):
    prediction = predict(input_data)
    
    if prediction == 1:
        st.success('The customer is likely to churn.')
    else:
        st.success('The customer is not likely to churn.')


import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler ,OneHotEncoder
import pickle

# Load the trained model
model = tf.keras.models.load_model('ann_model.h5')

# Load the scaler and label encoders
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('lable_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

#streamlit app title
st.title("Customer Churn Prediction")

# Input fields for user data
geography = st.selectbox("Geography",onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender",label_encoder_gender.classes_)
age = st.slider("Age", min_value=18, max_value=100)
credit_score = st.number_input("Credit Score", min_value=0, max_value=850)
tenure = st.number_input("Tenure (in years)",0, 10)
balance = st.number_input("Balance")
num_of_products = st.number_input("Number of Products", 1, 4)
has_credit_card = st.selectbox("Has Credit Card", ["0", "1"])
is_active_member = st.selectbox("Is Active Member", ["0", "1"])
estimated_salary = st.number_input("Estimated Salary", min_value=0, max_value=1000000)

#prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

#one-hot encode the categorical features
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#concatenate the encoded features with the rest of the input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#scale the input data
input_data_scaled = scaler.transform(input_data)

#predict the output
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]  # Convert the prediction to a binary outcome
st.write(f"Prediction Probability: {prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.write("The customer is likely to leave the bank.")
else:
    st.write("The customer is likely to stay with the bank.")


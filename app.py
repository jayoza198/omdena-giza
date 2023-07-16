import streamlit as st
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv("/content/drive/MyDrive/CSV files/Omdena-Giza/tweet_data_clean.csv")

# Drop the irrelevant columns
df = df.drop(["Unnamed: 0", "type_of_disaster", "hashtags"], axis=1)

# Create the features
X = df["tweet_text"]
y = df["disaster"]

# Remove the spaces from the strings
X = X.str.strip()

# Scale the data
scaler = preprocessing.StandardScaler()

X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25)

# Create the model
model = LogisticRegression(max_iter=1000)

# Fit the model to the training data
model.fit(X_train, y_train)

# Create a function to predict whether a tweet is a disaster
def predict_disaster(text):
    # Create the features
    X = [text]

    # Scale the data
    X_scaled = scaler.transform(X)

    # Predict the label
    prediction = model.predict(X_scaled)

    return prediction[0]

# Title
st.title("Tweet Disaster Prediction")

# Input text
text = st.text_input("Enter text: ")

# Predict
if st.button("Predict"):
    prediction = predict_disaster(text)
    if prediction == 1:
        st.write("The tweet is about a disaster.")
    else:
        st.write("The tweet is not about a disaster.")

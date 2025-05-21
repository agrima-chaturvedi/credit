import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
credit_card_data=pd.read_csv('creditcard.csv')
legit=credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data.Class==1]
#Under sampling
legit_sample=legit.sample(n=492,random_state=2)
#concatenate two data set
new_dataset=pd.concat([legit_sample,fraud],axis=0)
#Split data into features(X) and targets(Y)
X=new_dataset.drop(columns='Class',axis=1)
Y=new_dataset['Class']
#Split data into taining and testing data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
#Model Training
#Logistic Regression
model=LogisticRegression()
#training the Logistic Regression Model with Training data
model.fit(X_train,Y_train)
#Model Evaluation
#Accuracy Score
#accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
#accuracy on test data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)

#create Streamlit app
st.title("credit Fraud Detection Model")
input_df=st.text_input("Enter all Required features values")
input_df_splited=input_df.split(',')

submit=st.button("submit")
if submit:
    features=np.asarray(input_df_splited,dtype=np.float64)
    prediction=model.predict(features.reshape(1,-1))

    if prediction[0]==0:
        st.write("Legitimate Transaction")
    else:
        st.write("Fradulant Transaction")
    
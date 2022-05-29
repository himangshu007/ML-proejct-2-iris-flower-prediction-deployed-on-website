from matplotlib.style import use
import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write(
    """
    # Iris Flower Prediction App

    This app predicts the Iris Flower type!
    """
)
st.image('iris-dataset.png')
st.sidebar.header('User Input Parameters')




def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length',4.3,7.9,5.4)
    sepal_width = st.sidebar.slider('Sepal Width',2.0,4.4,3.4)
    petal_length = st.sidebar.slider('Petal Length',1.0,6.9,1.3)
    petal_width = st.sidebar.slider('Petal Width',0.1,2.5,0.2)
    data ={
        'Sepal Length':sepal_length,
        'Sepal Width':sepal_width,
        'Petal Length':petal_length,
        'Petal Width':petal_width
    }
    features= pd.DataFrame(data,index=[0])
    return features

df = user_input_features()

st.subheader('User Input Interface')
st.write(df)

iris = datasets.load_iris()
x=iris.data
y=iris.target

clf = RandomForestClassifier()
clf.fit(x,y)

pred = clf.predict(df)
pred_prob = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[pred])

if iris.target_names[pred] == 'setosa':
    st.image('setosa.jpg')
if iris.target_names[pred] =='versicolor':
    st.image('versicolor.jpg')
if iris.target_names[pred] =='virginica':
    st.image('virginica.jpg')

st.subheader('Prediction Probability')
st.write(pred_prob)

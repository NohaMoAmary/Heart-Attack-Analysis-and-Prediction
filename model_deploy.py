# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 19:09:15 2022

@author: noham
"""

import numpy as np
import pickle
import pandas as pd

import streamlit as st 



pickle_in=open('model_svm.pkl','rb')

# will use this model SVM to make new predictions 
svm_model =pickle.load(pickle_in)


# load scaler 
pickle_in = open ('scaler.pkl','rb')
scaler = pickle.load(pickle_in)


#now will create a function
# input data are the features user will insert  
def HeartAttak(input_data):
    
    input_data = scaler.transform(input_data)
    
    input_data_numpy=input_data.reshape(1,-1)
    
    result = svm_model.predict (input_data_numpy)
    
    if (result[0]==0):
      return( "no heart attake")
    else :
      return('possibility of heart attack')

def main():
    
    # title for the app
    st.title("Heart Attack Analysis & Prediction")
    
    #get input from user 
   
    				
    age = st.text_input('your Age ')
    sex = st.text_input('your sex ')
    ChestPaintype = st.text_input('ChestPaintype')
    RestingBloodPressure = st.text_input('RestingBloodPressure')
    chol = st.text_input('cholestrol level ')
    FastingBloodSugar = st.text_input('FastingBloodSugar')
    restecg = st.text_input('restecg')
    MaxHeartRate = st.text_input('MaxHeartRate')
    ExInducedAngina = st.text_input('ExInducedAngina')
    oldpeak = st.text_input('oldpeak')
    slope = st.text_input('slope')
    MajorVessels = st.text_input('MajorVessels')
    ThaliumStressResult = st.text_input('ThaliumStressResult')
    o2Saturation = st.text_input('o2Saturation')
    
    # code for prediction 
    Result = ''
    
    #button for prediction 
    if st.button("Result"):
        Result=HeartAttak([[age,sex,ChestPaintype,RestingBloodPressure,chol,FastingBloodSugar,restecg,MaxHeartRate,ExInducedAngina,oldpeak,slope,MajorVessels,ThaliumStressResult,o2Saturation]])
    
    st.success('The output is {}'.format(Result))
    

if __name__=='__main__':
    main()


















    
    
    
    
    
    
    
    
    
    
    
    
    
    
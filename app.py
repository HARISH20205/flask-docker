from flask import Flask,request
import numpy as np
import pandas as pd
import pickle

app=Flask(__name__)
pin=open('classifier.pkl','rb')
classifier=pickle.load(pin)
@app.route("/")
def welcome():
    return "Welcome all!"

@app.route('/predict')
def display():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The Predicted Value is : "+str(prediction)

@app.route('/predict_file',methods=['POST'])
def predict():
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return "The Predicted Values for the csv is "+  str(list(prediction))

if __name__=="__main__":
    app.run()




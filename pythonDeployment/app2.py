from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import json,numpy
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))


app = Flask(__name__)
@app.route('/pred/', methods=['POST'])
def makecalc():
    scaler=MinMaxScaler(feature_range=(0,1))
    print(request.json)
    #data = request.json()
    data = request.get_json()
    X = np.asarray(data)
    X_test = scaler.fit_transform(np.array(X).reshape(-1,1))
    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_test =X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    output = model1.predict(X_test)
    pred  = scaler.inverse_transform(output)
    prediction = np.array2string(pred)
    print(output)
    #Uncomment to return JSON OUTPUT
    #return jsonify(prediction)
    return prediction

if __name__ == '__main__':
    model1 = load_model('my_model2.h5')
    app.run(debug=True)
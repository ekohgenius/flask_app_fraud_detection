import ast

from flask import Flask, jsonify, request

import json
import joblib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

app = Flask(__name__)

criterion = nn.MSELoss()

def test_model(loader, model):

    """
    This function evaluate the model on the test data
    """
      
    model.eval()
    losses = []
    
     # enumerate mini batches
    for inputs in loader:
        inputs = inputs
           
        # compute the model output
        out = model(inputs)
        
        # calculate loss
        loss = criterion(out, inputs)

        losses += [loss.item()]
        
    return losses


def get_prediction(json_object: dict)-> json:

    """
    This function takes in a json file, performs predictions on it and returns a json prediction
    Args:
        file: json object containing input features

    Return:
        json object having predictions 
    """

    
    data = pd.DataFrame(json_object, index=[0])

    prediction = {'isFraud': False}

    if (data.type == "CASH_OUT").bool() or (data.type == "TRANSFER").bool():
        target_encoding = joblib.load('target_encoding.gz')
        min_max_scaler = joblib.load('min_max_scaler.gz')
        saved_model = torch.load('model.pth')

        X = target_encoding.transform(data)
        X = min_max_scaler.transform(X)

        test_data = torch.FloatTensor(X)
        test_args = dict(shuffle = False, batch_size = 1, num_workers=1)
        test_loader = DataLoader(test_data, **test_args)

        reconstruction_loss_test = test_model(test_loader, saved_model)

        if np.array(reconstruction_loss_test) > 0.001:
            prediction['isFraud'] = True

    
    json_object = json.dumps(prediction)

    return json_object

    


## Flask API




@app.route('/is-fraud', methods=['POST'])
def predict():
        # we will get the file from the request
        file = request.json
     
        
        json_object = get_prediction(file)
        print(json_object)

        return json_object
       

if __name__ == '__main__':
     app.run(debug=True ,port=8080,use_reloader=False)
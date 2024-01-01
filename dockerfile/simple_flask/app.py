from flask import Flask, request
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import json
import logging
from paul_sentiment_analysis.tokenize_data import tokenize_text

app = Flask(__name__)

model = tf.keras.models.load_model('/models/lstm.h5', compile=False)
vectorizer = tf.keras.models.load_model('/models/tf_vectorizer')

@app.post("/predict")
def run_ai():
    if request.headers['aikey'] == os.environ['AIKEY']:
        if request.is_json:
            try:
                data = request.get_json()['json_data']
                logging.info('successfully retrieved data')
                data = pd.DataFrame.from_dict(json.loads(data))
                data['0'] = data['0'].apply(tokenize_text)
                logging.info('successfully converted data to data frame')
                predictions = model(vectorizer(data)).numpy().tolist()
                logging.info('successfully ran prediction models')
                return predictions, 201
            except Exception as e:
                return {"error": str(e)}, 400
        else:
            return {"error": "Request must be JSON"}, 415
    else:
        return {'error': 'Wrong key'}, 401

if __name__ == "__main__":
    app.run()
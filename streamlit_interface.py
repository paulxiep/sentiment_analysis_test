import streamlit as st
import pandas as pd
import requests
import numpy as np
import seaborn as sns
import environ
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

def predict(review, model_choice):
    st.markdown('ran flask api')
    headers = {
        'aikey': environ.Env()('AIKEY')
    }
    if 'csv' not in review:
        json_body = {
            'json_data': pd.DataFrame([[review_text]]).to_json(),
            'model_choice': model_choice
        }
        labels = None
    else:
        data = pd.read_csv(review)
        data.columns = [i for i in range(len(data.columns))]
        json_body = {
            'json_data': data[[0]].to_json()
        }
        if 1 in data.columns:
            labels = data[1]
        else:
            labels = None
    host = f"http://localhost:8089"
    response = requests.post(f"{host}/predict",
                             headers=headers,
                             json=json_body).json()

    return response, labels


def display_predictions(predictions, labels):
    probabilities = pd.DataFrame.from_dict(predictions).rename(columns={
        0: 'prob_negative',
        1: 'prob_neutral',
        2: 'prob_positive'
    })
    probabilities['prediction'] = np.vectorize(lambda x: {0: 'neg', 1: 'neu', 2: 'pos'}[x]
                                            )(np.argmax(np.array(probabilities), axis=1))
    if labels is not None:
        probabilities['label'] = np.array(labels)
        st.write(f"accuracy: {accuracy_score(probabilities['prediction'], probabilities['label'])}")
        st.write(f"f1: {f1_score(probabilities['prediction'], probabilities['label'], average='weighted')}")
        confusion = confusion_matrix(probabilities['prediction'], probabilities['label'])
        st.pyplot(sns.heatmap(confusion, annot=True, fmt='d',
                              xticklabels=['neg', 'neu', 'pos'], yticklabels=['neg', 'neu', 'pos']).get_figure())
    st.dataframe(probabilities)


model_choice = st.radio('choose model', options=['LSTM'])
review_text = st.text_input('Input review text in Thai')
predict_button = st.button('predict')


if predict_button:
    display_predictions(*predict(review_text, model_choice))

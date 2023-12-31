import os
import re

import pandas as pd
from pythainlp import word_tokenize, sent_tokenize


def filter_thai(text):
    '''
    basically, filter out special characters
    '''
    pattern = re.compile(r"[^\u0E00-\u0E7F ]|^'|'$|''")
    char_to_remove = re.findall(pattern, text)
    list_with_char_removed = [char for char in text if not char in char_to_remove]
    return ''.join(list_with_char_removed)


def read_raw_data():
    '''
    read all the saved raw data from data scraping modules,
    rename columns accordingly
    '''
    return pd.concat(
        [pd.read_csv(os.path.join('raw_data', path)).assign(type=path.split('_')[-3]).rename(
            columns={'0': 'review', '1': 'rating'}) for path in os.listdir('raw_data') if path[-3:] == 'csv'],
        axis=0
    )


def score_to_sentiment(data):
    '''
    map review score 5 to label 2 (positive)
    score 4 to label 1 (neutral)
    and score 1-3 to label 0 (negative)
    '''
    data['rating'] = data['rating'].apply(lambda x: max(0, x - 3))
    return data


def tokenize_data(x):
    return ' '.join(list(filter(lambda y: y.replace(' ', ''), word_tokenize(filter_thai(x)))))


# def tokenize_data_sentence(x):
#     return [tokenize_data(y) for y in sent_tokenize(filter_thai(x))]


def create_sklearn_ngram_tokenizer(data, ngram_range=(1, 2), min_df=20):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(tokenizer=tokenize_data,
                            ngram_range=ngram_range,
                            min_df=min_df, sublinear_tf=True)
    return tfidf.fit(data)


def sklearn_ngram_tokenize(data, sklearn_ngram_tokenizer):
    return sklearn_ngram_tokenizer.fit(data).toarray()


def create_tensorflow_vectorize_layer(x_train, vocab_size=512, max_tokens=128):
    from tensorflow.keras.layers import TextVectorization
    vectorize_layer = TextVectorization(
        standardize=None,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=max_tokens)
    vectorize_layer.adapt(x_train)
    return vectorize_layer


def tensorflow_tokenize(data, vectorize_layer):
    '''
    expects input data to be sentence_tokenized tensor
    returns data as tokenized by vectorize_layer
    '''
    import tensorflow as tf
    # print(data)
    return tf.map_fn(lambda x: vectorize_layer(x), data, dtype=tf.int64)
    # data = data.apply(lambda x: ' '.join(tokenize_data(x)))
    # return vectorize_layer(data)


# def compile_unique_tokens(data):
#     '''
#     compile all unique tokens in the data
#     '''
#     return reduce(lambda a, b: a.union(set(b)), list(data['tokenized'].apply(lambda x: set(x))), set([]))


# if __name__ == '__main__':
#     token_set = compile_unique_tokens(tokenize_data(read_raw_data()))
#     print(token_set)
#     print(len(token_set))

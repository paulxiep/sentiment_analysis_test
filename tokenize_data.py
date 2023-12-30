import os
import re
from functools import reduce

import pandas as pd
from pythainlp import word_tokenize


def filter_thai(text):
    '''
    basically, filter out special characters
    '''
    pattern = re.compile(r"[^\u0E00-\u0E7Fa-zA-Z ]|^'|'$|''")
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


def tokenize_data(data):
    '''
    tokenize review in data table, save as new column 'tokenized'
    '''
    return pd.concat([data, data['review'] \
                     .apply(lambda x: list(filter(lambda y: y.replace(' ', ''),
                                                  word_tokenize(filter_thai(x))))).rename('tokenized')], axis=1)


def compile_unique_tokens(data):
    '''
    compile all unique tokens in the data
    '''
    return reduce(lambda a, b: a.union(set(b)), list(data['tokenized'].apply(lambda x: set(x))), set([]))


if __name__ == '__main__':
    token_set = compile_unique_tokens(tokenize_data(read_raw_data()))
    print(token_set)
    print(len(token_set))

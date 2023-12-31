import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.data import Dataset

from tokenize_data import create_sklearn_ngram_tokenizer, create_tensorflow_vectorize_layer
from tokenize_data import score_to_sentiment, read_raw_data, tokenize_data
from tokenize_data import sklearn_ngram_tokenize, tensorflow_tokenize


def prepare_train_test(test_size=0.2):
    data = score_to_sentiment(read_raw_data().drop_duplicates(ignore_index=True))
    data = data[data['review'].apply(tokenize_data).apply(lambda x: len(x) > 0)]
    train, test = train_test_split(data, test_size=test_size, stratify=data['type'])
    return train, test, data


def upsample_train(train):
    return train.loc[train.index.repeat(((3 - train.rating) ** 1).astype(int))]


def prepare_tfidf_data(test_size=0.2, upsample=True):
    train, test, data = prepare_train_test(test_size)
    sklearn_ngram_tokenizer = create_sklearn_ngram_tokenizer(data['review'])
    if upsample:
        train = upsample_train(train)
    x_train = sklearn_ngram_tokenize(train['review'], sklearn_ngram_tokenizer)
    x_test = sklearn_ngram_tokenize(test['review'], sklearn_ngram_tokenizer)
    y_train, y_test = train['rating'], test['rating']
    return x_train, x_test, y_train, y_test


def random_review_slice(reviews):
    '''
    a data augmentation method I devised
    randomly cut the starting part of review,
    so long reviews exceeding token limit has a random chance of the later part to be read
    '''
    pos = tf.random.uniform(shape=[],
                            minval=0,
                            maxval=tf.math.maximum(tf.math.minimum(tf.strings.length(reviews)[-1] // 2,
                                                                   tf.strings.length(reviews)[-1] - 160), 1),
                            dtype=tf.int32)
    substr = tf.strings.substr(reviews, pos=pos, len=-1)
    splitted = tf.strings.split(substr, sep=' ')
    selected = splitted[:, tf.cast(tf.math.greater(tf.strings.length(reviews)[-1] - pos, 160), tf.int32):]

    return tf.strings.reduce_join(selected, separator=' ', axis=-1)


def prepare_lstm_datasets(test_size=0.2, upsample=True, random_slice_string_data=True,
                          vocab_size=256, max_tokens=64):
    train, test, data = prepare_train_test(test_size)
    tensorflow_vectorize_layer = create_tensorflow_vectorize_layer(train['review'].apply(tokenize_data),
                                                                   vocab_size=vocab_size, max_tokens=max_tokens)

    train['review'] = train['review'].apply(tokenize_data)
    test['review'] = test['review'].apply(tokenize_data)

    if upsample:
        train = upsample_train(train)

    if random_slice_string_data:
        train = Dataset.from_tensor_slices((train[['review']], train[['rating']])) \
            .map(lambda x, y: (
        tensorflow_tokenize(random_review_slice(x), tensorflow_vectorize_layer), tf.one_hot(y, depth=3, axis=-1)))
    else:
        train = Dataset.from_tensor_slices((train[['review']], train[['rating']])) \
            .map(lambda x, y: (
        tensorflow_tokenize(x, tensorflow_vectorize_layer), tf.one_hot(y, depth=3, axis=-1)))

    test = Dataset.from_tensor_slices((test[['review']], test[['rating']])) \
        .map(lambda x, y: (tensorflow_tokenize(x, tensorflow_vectorize_layer), tf.one_hot(y, depth=3, axis=-1)))

    return train, test

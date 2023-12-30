import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization, LSTM, Dense, Input, Embedding, Bidirectional, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tokenize_data import score_to_sentiment, tokenize_data, read_raw_data

data = score_to_sentiment(tokenize_data(read_raw_data().drop_duplicates(ignore_index=True)))
data['tokenized'] = data['tokenized'].apply(lambda x: ' '.join(x))
x_train, x_test, y_train, y_test = train_test_split(data[['tokenized']], data['rating'], test_size=0.2,
                                                    stratify=data['type'] + data['rating'].astype(str))

vocab_size = 16384
max_tokens = 128
vectorize_layer = TextVectorization(
    standardize=None,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=max_tokens)
vectorize_layer.adapt(tf.expand_dims(x_train, -1))

x_train, x_test = vectorize_layer(x_train), vectorize_layer(x_test)
input = Input(max_tokens)

vectorized = Embedding(vocab_size + 1, 128, input_length=max_tokens)(input)

output = Dense(3, activation='softmax')(Dropout(0.2)((Bidirectional(layer=LSTM(128))(vectorized))))

model = Model(inputs=input, outputs=output)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print(tf.config.list_physical_devices('GPU'))
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=16, verbose=1)

from keras import Input, Model
from keras.layers import Embedding, CuDNNLSTM, Reshape, Concatenate, Dense


def text_location_model(text_vocab_size, location_vocab_size, text_embedding_size=8, text_hidden_size=300,
                        location_embedding_size=8):
    text = Input(shape=(None,))
    location = Input(shape=(1,))

    text_embedding = Embedding(text_vocab_size, text_embedding_size)(text)
    text_hidden = CuDNNLSTM(text_hidden_size)(text_embedding)

    location_embedding = Embedding(location_vocab_size, location_embedding_size)(location)
    location_embedding = Reshape((location_embedding_size,))(location_embedding)

    features = Concatenate()([text_hidden, location_embedding])
    prediction = Dense(1, activation='sigmoid')(features)

    model = Model(inputs=[text, location], outputs=prediction)
    return model
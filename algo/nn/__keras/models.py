import keras.backend as K
from keras import Input, Model
from keras.initializers import glorot_normal, orthogonal
from keras.layers import Embedding, SpatialDropout1D, Bidirectional, GRU, Flatten, Dense, Dropout, BatchNormalization, \
    Reshape, Conv2D, MaxPool2D, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, LSTM, Conv1D, \
    Activation
from keras.optimizers import Adam

from algo.nn.__keras.layers import Capsule, Attention
from algo.nn.__keras.utility import f1_smart
from algo.nn.__keras.wrappers import DropConnect


def capsule(maxlen, max_features, embed_size, embedding_matrix, num_classes):
    K.clear_session()
    inp = Input(shape=(maxlen,))

    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)

    x = SpatialDropout1D(rate=0.2)(x)
    x = Bidirectional(GRU(100, return_sequences=True,
                          kernel_initializer=glorot_normal(seed=12300),
                          recurrent_initializer=orthogonal(gain=1.0, seed=10000)))(x)

    x = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(x)
    x = Flatten()(x)

    x = Dense(100, activation="relu", kernel_initializer=glorot_normal(seed=12300))(x)
    x = Dropout(0.12)(x)
    x = BatchNormalization()(x)

    x = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model


def cnn_2d(maxlen, max_features, embed_size, embedding_matrix, num_classes):
    K.clear_session()

    filter_sizes = [1, 2, 3, 5]
    num_filters = 32

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.4)(x)
    x = Reshape((maxlen, embed_size, 1))(x)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size), kernel_initializer='normal',
                    activation='elu')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size), kernel_initializer='normal',
                    activation='elu')(x)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), kernel_initializer='normal',
                    activation='elu')(x)
    conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size), kernel_initializer='normal',
                    activation='elu')(x)

    maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1))(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1))(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1))(conv_2)
    maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] + 1, 1))(conv_3)

    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(num_classes, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def pooled_gru(maxlen, max_features, embed_size, embedding_matrix, num_classes):
    K.clear_session()
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(num_classes, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def lstm_attention(maxlen, max_features, embed_size, embedding_matrix, num_classes):
    K.clear_session()
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(256, activation="relu")(x)
    # x = Dropout(0.25)(x)
    x = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def lstm_gru_attention(maxlen, max_features, embed_size, embedding_matrix, num_classes):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(LSTM(40, return_sequences=True))(x)
    y = Bidirectional(GRU(40, return_sequences=True))(x)

    atten_1 = Attention(maxlen)(x)  # skip connect
    atten_2 = Attention(maxlen)(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)

    conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
    conc = Dense(16, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(num_classes, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def attention_capsule(maxlen, max_features, embed_size, embedding_matrix, num_classes):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(rate=0.24)(x)
    x = Bidirectional(LSTM(80,
                                return_sequences=True,
                                kernel_initializer=glorot_normal(seed=1029),
                                recurrent_initializer=orthogonal(gain=1.0, seed=1029)))(x)

    x_1 = Attention(maxlen)(x)
    x_1 = DropConnect(Dense(32, activation="relu"), prob=0.2)(x_1)

    x_2 = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(x)
    x_2 = Flatten()(x_2)
    x_2 = DropConnect(Dense(32, activation="relu"), prob=0.2)(x_2)
    conc = concatenate([x_1, x_2])
    # conc = add([x_1, x_2])
    outp = Dense(num_classes, activation="sigmoid")(conc)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model



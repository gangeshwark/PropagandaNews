from keras.layers import Conv2D, MaxPool2D
from keras.layers import Embedding, Concatenate
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *


def model_CNN(embeddingMatrix):
    x1 = Input(shape=(140,), name='main_input1')
    embeddingLayer = Embedding(input_dim=embeddingMatrix.shape[0], output_dim=300, weights=[embeddingMatrix],
                               mask_zero=True, input_length=140, trainable=True)
    emb1 = embeddingLayer(x1)
    emb1 = Activation('tanh')(emb1)
    expand_dim = Lambda(lambda x: K.expand_dims(x, axis=-1))
    emb1 = expand_dim(emb1)
    flatten = Flatten()
    conv_0 = Conv2D(128, kernel_size=(2, 300), padding='valid', activation='relu')
    conv_1 = Conv2D(128, kernel_size=(3, 300), padding='valid', activation='relu')
    conv_2 = Conv2D(128, kernel_size=(4, 300), padding='valid', activation='relu')
    maxpool_0 = MaxPool2D(pool_size=(99, 1), strides=(1, 1), padding='valid')
    maxpool_1 = MaxPool2D(pool_size=(98, 1), strides=(1, 1), padding='valid')
    maxpool_2 = MaxPool2D(pool_size=(97, 1), strides=(1, 1), padding='valid')

    activation = Activation('relu')
    conv_0_e1 = conv_0(emb1)
    conv_1_e1 = conv_1(emb1)
    conv_2_e1 = conv_2(emb1)
    maxpool_0_e1 = maxpool_0(conv_0_e1)
    maxpool_1_e1 = maxpool_1(conv_1_e1)
    maxpool_2_e1 = maxpool_2(conv_2_e1)
    out_0_e1 = flatten(maxpool_0_e1)
    out_1_e1 = flatten(maxpool_1_e1)
    out_2_e1 = flatten(maxpool_2_e1)
    e1 = Concatenate(axis=-1)([out_0_e1, out_1_e1, out_2_e1])
    # expand_dim = Lambda(lambda x: K.expand_dims(x, axis=-2))
    e1 = activation(e1)
    # e1 = batch_norm1(e1)
    # e2 = batch_norm2(e2)
    # e3 = batch_norm3(e3)
    # e = expand_dim(e1)
    # e = Reshape((3, 384))(e)
    # lstm_1 = LSTM(300, return_sequences=True)(e)
    # squeeze = Lambda(lambda x: K.squeeze(x, axis=1))
    # lstm_1 = squeeze(lstm_1)

    att = Dense(600, activation='relu')(e1)
    out = Dense(14, activation='softmax')(att)
    model = Model([x1], out)
    print(model.summary())
    return model


def model_CNN_emo(embeddingMatrix):
    x = Input(shape=(140,), name='text_input')
    x_emo = Input(shape=(7,), name='emo_input')
    embeddingLayer = Embedding(input_dim=embeddingMatrix.shape[0], output_dim=300, weights=[embeddingMatrix],
                               mask_zero=True, input_length=140, trainable=True)
    emb1 = embeddingLayer(x)
    emb1 = Activation('tanh')(emb1)
    expand_dim = Lambda(lambda x: K.expand_dims(x, axis=-1))
    emb1 = expand_dim(emb1)
    flatten = Flatten()
    conv_0 = Conv2D(128, kernel_size=(2, 300), padding='valid', activation='relu')
    conv_1 = Conv2D(128, kernel_size=(3, 300), padding='valid', activation='relu')
    conv_2 = Conv2D(128, kernel_size=(4, 300), padding='valid', activation='relu')
    maxpool_0 = MaxPool2D(pool_size=(99, 1), strides=(1, 1), padding='valid')
    maxpool_1 = MaxPool2D(pool_size=(98, 1), strides=(1, 1), padding='valid')
    maxpool_2 = MaxPool2D(pool_size=(97, 1), strides=(1, 1), padding='valid')

    activation = Activation('relu')
    conv_0_e1 = conv_0(emb1)
    conv_1_e1 = conv_1(emb1)
    conv_2_e1 = conv_2(emb1)
    maxpool_0_e1 = maxpool_0(conv_0_e1)
    maxpool_1_e1 = maxpool_1(conv_1_e1)
    maxpool_2_e1 = maxpool_2(conv_2_e1)
    out_0_e1 = flatten(maxpool_0_e1)
    out_1_e1 = flatten(maxpool_1_e1)
    out_2_e1 = flatten(maxpool_2_e1)
    e1 = Concatenate(axis=-1)([out_0_e1, out_1_e1, out_2_e1])
    # expand_dim = Lambda(lambda x: K.expand_dims(x, axis=-2))
    e1 = activation(e1)
    # e1 = batch_norm1(e1)
    # e2 = batch_norm2(e2)
    # e3 = batch_norm3(e3)
    # e = expand_dim(e1)
    # e = Reshape((3, 384))(e)
    # lstm_1 = LSTM(300, return_sequences=True)(e)
    # squeeze = Lambda(lambda x: K.squeeze(x, axis=1))
    # lstm_1 = squeeze(lstm_1)

    att = Dense(300, activation='relu')(e1)

    # emo_layer = Dense(10, activation='relu')(x_emo)
    concat_layer = Concatenate(axis=-1)([att, x_emo])
    final_layer = Dense(100, activation='relu')(concat_layer)

    out = Dense(14, activation='softmax')(final_layer)
    model = Model([x, x_emo], out)
    print(model.summary())
    return model


def model_LSTM(embeddingMatrix):
    x1 = Input(shape=(140,), name='main_input1')
    embeddingLayer = Embedding(input_dim=embeddingMatrix.shape[0], output_dim=300, weights=[embeddingMatrix],
                               mask_zero=True, input_length=140, trainable=True)
    emb1 = embeddingLayer(x1)
    emb1 = Activation('tanh')(emb1)
    lstm_1 = LSTM(100, return_sequences=False)(emb1)

    att = Dense(100, activation='relu')(lstm_1)
    out = Dense(14, activation='softmax')(att)
    model = Model([x1], out)
    print(model.summary())
    return model

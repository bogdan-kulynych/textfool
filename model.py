import spacy
import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras import backend as K


def get_embeddings(vocab):
    max_rank = max(lex.rank+1 for lex in vocab if lex.has_vector)
    vectors = np.ndarray((max_rank+1, vocab.vectors_length), dtype='float32')
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank + 1] = lex.vector
    return vectors


vocab_nlp = spacy.load('en', parser=False, tagger=False, entity=False)
print('Preparing embeddings...')
embeddings = get_embeddings(vocab_nlp.vocab)


def build_model(max_length=1000,
                nb_filters=64,
                kernel_size=3,
                pool_size=2,
                regularization=0.01,
                weight_constraint=2.,
                dropout_prob=0.4,
                clear_session=True):
    if clear_session:
        K.clear_session()

    model = Sequential()
    model.add(Embedding(
        embeddings.shape[0],
        embeddings.shape[1],
        input_length=max_length,
        trainable=False,
        weights=[embeddings]))

    model.add(Conv1D(nb_filters, kernel_size, activation='relu'))
    model.add(Conv1D(nb_filters, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size))

    model.add(Dropout(dropout_prob))

    model.add(Conv1D(nb_filters * 2, kernel_size, activation='relu'))
    model.add(Conv1D(nb_filters * 2, kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size))

    model.add(Dropout(dropout_prob))

    model.add(GlobalAveragePooling1D())
    model.add(Dense(1,
        kernel_regularizer=l2(regularization),
        kernel_constraint=maxnorm(weight_constraint),
        activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

    return model


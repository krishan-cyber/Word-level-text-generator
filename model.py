#import necessary library pandas for data exploration,numpy for multidimensional array,tensorflow google's deep learning library
import pandas as pd
import numpy as np
import string, os
import tensorflow as tf
from numpy.random import seed
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku

tokenizer = Tokenizer()
corpus=['working fine','delhi is good place','language is just way to communicate','seems good']
#convert words into sequences for feeding into model
def get_sequence_of_tokens(corpus):
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words
inp_sequences, total_words = get_sequence_of_tokens(corpus)


#pad generated sequences to insure same length sequeneces easy to train for tensorflow model
def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    X_train, Y_train = input_sequences[:,:-1],input_sequences[:,-1]
    Y_train = ku.to_categorical(Y_train, num_classes=total_words)
    return X_train, Y_train, max_sequence_len
X_train, Y_train, max_sequence_len = generate_padded_sequences(inp_sequences)

#create model added lstm layer for improved predictions
def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=input_len))
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
model = create_model(max_sequence_len, total_words)
model.summary()
model.fit(X_train,Y_train, epochs=100)

#getting model response
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list],maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list)
        predicted=np.argmax(predicted,axis=1)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()
print (generate_text("delhi", 3, model, max_sequence_len))















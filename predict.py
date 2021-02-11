import deep_learning.hpsearch
import pickle
import numpy as np
import tensorflow.compat.v1.logging as log
from deep_learning.model import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def model_init():
    ''' Retrieve the neural network model and load the relevant parameter '''

    # Parameter initialization
    total_words, embedding_dim, max_length, dropout_factor, num_epochs = deep_learning.hpsearch.get_hyperparameters('deep_learning/model_params.txt')
    weights = 'deep_learning/cnn-blstm-weights.hdf5'
    tokenizer_path = 'deep_learning/tokenizer.pickle'

    # Load the model
    model = Model(total_words=total_words,
                  embedding_dim=embedding_dim,
                  max_length=max_length,
                  dropout_factor=dropout_factor)
    model.build()
    model.load(weights)
    model.compile()

    # Load the tokens used during training
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    return model, tokenizer, max_length

def predict_sentiment(text):
    ''' Predict the sentiment of a single string '''

    model, tokenizer, max_length = model_init()

    # Convert the text to a numeric sequence and pad with zeros on the right
    data_seq = tokenizer.texts_to_sequences([text])
    input_data = np.array(pad_sequences(data_seq, padding='post', maxlen=max_length))

    return model.predict(input_data)[0][1]

def main():
    log.set_verbosity(log.ERROR)
    prediction = predict_sentiment('This is a test message')
    print('Sentiment is ' + (prediction and 'negative' or 'positive'))

if __name__ == '__main__':
    main()

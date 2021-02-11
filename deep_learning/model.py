import numpy as np
from tensorflow.keras.layers import Embedding, Flatten, Conv1D, LSTM, Dense, Bidirectional, MaxPooling1D, Dropout
from tensorflow.keras.models import Sequential
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

class Model:
    '''A class to make the main (training) method slightly simpler and more organized'''
    
    def __init__(self, x_train=[], y_train=[], x_test=[], y_test=[], val_split=0,
                 total_words=0, embedding_dim=0, max_length=0, dropout_factor=0,
                 num_epochs=0, batch_size=0, output_dir=''):
        '''Establish the model parameters as instance variables so that they are only inputted once
        Parameters are optional so that the model can also be loaded easily
        '''

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.val_split = val_split
        self.total_words = total_words
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.dropout_factor = dropout_factor
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.callbacks = []

    def build(self):
        '''Set up the deep neural network model'''

        self.model = Sequential()

        self.model.add(Embedding(self.total_words, self.embedding_dim, input_length=self.max_length, name='Embed'))
        self.model.add(Dropout(rate=self.dropout_factor, name='Dropout1'))
        self.model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu', name='Conv1'))
        self.model.add(MaxPooling1D(pool_size=2, name='MaxPoo11'))
        self.model.add(Dropout(rate=self.dropout_factor, name='Dropout2'))
        self.model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='Conv2'))
        self.model.add(MaxPooling1D(pool_size=2, name='MaxPoo12'))
        self.model.add(Dropout(rate=self.dropout_factor, name='Dropout3'))
        self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', name='Conv3'))
        self.model.add(MaxPooling1D(pool_size=2, name='MaxPoo13'))
        self.model.add(Dropout(rate=self.dropout_factor, name='Dropout4'))
        self.model.add(Bidirectional(LSTM(units=32, dropout=self.dropout_factor, name='LSTM'), name='B-LSTM'))
        self.model.add(Dense(1, activation='sigmoid', name='Output'))

        return self.model  # Return is only needed when loading the model for testing

    def summary(self):
        '''Print each layer of the model with their output shapes'''
        
        self.model.summary()

    def save_summary(self):
        '''Save the model summary to a text file'''

        with open(self.output_dir+'/model.txt', 'w') as f:
            self.model.summary(print_fn=(lambda s: f.write(s + '\n')))

    def save_test_sequences(self):
        '''Save the test data so that the model can be tested on new data that it was not trained on'''
        
        np.save(self.output_dir+'/x_test.npy', self.x_test)
        np.save(self.output_dir+'/y_test.npy', self.y_test)

    def save_hyperparameters(self, output_dir):
        '''Write the relevant hyperparameters to a file so they can be retrieved after training'''
        with open(output_dir+'/model_params.txt', 'w') as f:
            f.write(str(self.total_words) + '\n' + \
                    str(self.embedding_dim) + '\n' + \
                    str(self.max_length) + '\n' + \
                    str(self.dropout_factor) + '\n' + \
                    str(self.num_epochs))

    def set_checkpoint_callback(self, param):
        '''Set a checkpoint to the model's callbacks so that weights can be loaded later'''
        
        checkpoint = ModelCheckpoint(self.output_dir+'/ep{epoch:02d}-%s{%s:.2f}.hdf5'%(param,param),
                                     monitor='val_accuracy',
                                     save_best_only=False)
        self.callbacks.append(checkpoint)

    def set_log_callback(self):
        '''Set a logger to write the accuracies and losses in each epic to a file'''
        
        csv_logger = CSVLogger(self.output_dir+'/training.log')
        self.callbacks.append(csv_logger)

    def compile(self, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
        '''Compile the model with default settings for simplicity'''

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
    def train(self):
        '''Train the model and return its history to be used if needed'''
        
        return self.model.fit(self.x_train,
                              self.y_train,
                              batch_size=self.batch_size,
                              epochs=self.num_epochs,
                              validation_split=self.val_split,
                              callbacks=self.callbacks)

    def load(self, weights_path):
        self.model.load_weights(weights_path)

    def test(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=2)

    def predict(self, data):
        probabilities = np.array(self.model.predict(data))
        predictions = np.array(self.model.predict_classes(data))
        return np.hstack((probabilities, predictions))

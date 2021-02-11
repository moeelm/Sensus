import load_data
import metrics
from model import Model

import os
from time import time, strftime, localtime


def main():
    # Hyper-parameters (to be tuned)
    num_epochs = 30
    embedding_dim = 32
    val_split = 0.176  # 15% of total data will be in validation set
    test_split = 0.15
    batch_sz = 32
    dropout_factor = 0.5
    params = ['accuracy', 'loss']

    # Set up the output directory for the weights and accuracies to go
    output_dir = 'Training_' + strftime('%Y_%m_%d_%Hh%Mm%Ss', localtime(time()))
    os.mkdir(output_dir)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # Load the data from the .csv file
    train_data, train_labels, test_data, test_labels, max_length, total_words, train_num, val_num = \
        load_data.load_data('./sentences_full.csv',
                            'text',
                            'sentiment',
                            ['negative', 'positve'],  # positve = 1, negative = 0
                            test_split,
                            val_split)
    
    # Initialize model parameters
    model = Model(train_data,
                  train_labels,
                  test_data,
                  test_labels,
                  val_split,
                  total_words+1,  # Add one for out-of-vocabulary
                  embedding_dim,
                  max_length,
                  dropout_factor,
                  num_epochs,
                  batch_sz,
                  output_dir)

    model.build()
    model.set_checkpoint_callback('val_accuracy')  # Change to 'val_acc' depending on Keras version
    model.set_log_callback()
    
    # Save the test data split in this run so that the model tests on different data than it was trained on
    model.save_test_sequences()

    # Save the model parameters so they can be used when loading the model
    model.save_hyperparameters(output_dir)

    # Print model and data information
    model.summary()
    model.save_summary()
    
    print('Training Sample Num: %d\n'\
          'Validation Sample Num: %d\n'\
          'Feature Size: %d\n'\
          'Saving results to: %s\n'
          % (train_num, val_num, max_length, output_dir))
    start = time()

    # Compile and train the model, saving the history logs so they can be reused later
    model.compile()
    history = model.train()

    end = time()
    print('Total time: %s s' % (end - start))
    metrics.plot(history, params, output_dir)

    

if __name__ == '__main__':
    main()

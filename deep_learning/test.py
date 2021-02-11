import hpsearch
import model
from time import time

def main():
    model_dir = 'Training_2021_02_08_15h10m20s\\'

    # Load the hyperparameters needed to build the model with the same layer shapes that wre used for training
    param_path = model_dir + 'model_params.txt'
    total_words, embedding_dim, max_length, dropout_factor, num_epochs = hpsearch.get_hyperparameters(param_path)

    weights_path = model_dir + hpsearch.get_weights_last(model_dir, num_epochs)
    
    # Construct the model and prepare it for testing
    trained = model.Model(total_words=total_words,
                          embedding_dim=embedding_dim,
                          max_length=max_length,
                          dropout_factor=dropout_factor)
    trained.build()
    trained.summary()
    trained.load(weights_path)
    trained.compile()

    # Load the test sequences that were split from the full dataset (separate from training and validation sequences)
    test_data, test_labels = hpsearch.get_test_sequences(model_dir)

    # Test and get the results
    start = time()
    loss, acc = trained.test(test_data, test_labels)[:2]
    end = time()
    print('Testing time: %s s' % (end - start,))
    print('Testing accuracy: %f' % (acc,))


if __name__ == '__main__':
    main()


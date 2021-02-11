import matplotlib.pyplot as plt

def plot(history, parameters, output_dir):
    '''Plot parameters such as loss and accuracy to see how they changed during the training'''
    
    i=1
    for param in parameters:
        plt.figure(i)
        plt.plot(history.history[param])  # Training phase
        plt.plot(history.history['val_'+param])  # Validation phase
        plt.title('Model '+param)
        plt.xlabel('Epochs')
        plt.ylabel(param)
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.grid()
        i += 1
        plt.savefig(output_dir+'/'+param)
    plt.show() 

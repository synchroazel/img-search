import matplotlib.pyplot as plt


def plot_history(history, model_name):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7.5))

    axes[0].plot(history.history['accuracy'])
    axes[0].plot(history.history['val_accuracy'])
    axes[0].set_title('Accuracy\n(Sparse Categorical Accuracy)')
    axes[0].set_ylabel('accuracy')
    axes[0].set_xlabel('epoch')
    axes[0].legend(['train', 'test'], loc='upper left')

    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'])
    axes[1].set_title('Loss\n(Sparse Categorical CrossEntropy)')
    axes[1].set_ylabel('loss')
    axes[1].set_xlabel('epoch')
    axes[1].legend(['train', 'test'], loc='upper left')

    fig.tight_layout()

    plt.savefig(f'reports/{model_name}_training.png')

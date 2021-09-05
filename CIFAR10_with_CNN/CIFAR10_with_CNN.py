import itertools
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout


# translator
LABELS =  {0: 'Airplane',
           1: 'Automobile',
           2: 'Bird',
           3: 'Cat',
           4: 'Deer',
           5: 'Dog',
           6: 'Frog',
           7: 'Horse',
           8: 'Ship',
           9: 'Truck'}


def load_data_():
    (x_train, y_train), (x_test, y_test) = load_data()

    print(f'x_train shape: {x_train.shape}')
    print(f'y_train shape: {y_train.shape}')

    print(f'Number of sample: {x_train.shape[0]}')
    print(f'Number of classes: {len(set(y_train.flatten()))}')

    x_train = x_train / 255
    x_test = x_test / 255

    return x_train, y_train, x_test, y_test

def build_model(input_shape, n_outputs):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', input_shape=(12,12,32)))
    model.add(Conv2D(filters=128, kernel_size=3, strides=2, activation='relu', input_shape=(12,12,64)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=n_outputs, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_loss(r):
    plt.title('Loss per iterations')
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.xlabel('iterations')
    plt.legend()
    plt.show()

def plot_acc(r):
    plt.title('Accuracy per iterations')
    plt.plot(r.history['accuracy'], label='accuracy')
    plt.plot(r.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('iterations')
    plt.legend()
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Cofusion Matrix without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i,j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_predictions(x_test, y_test, model):
    fig = plt.figure(figsize=(10, 10)) 
    for i in range(16):
        fig.add_subplot(4, 4, i+1) 
        idx = np.random.choice(x_test.shape[0])
        plt.imshow(x_test[idx]) 
        plt.axis('off')
        pre_label = LABELS[np.argmax(model.predict(x_test[idx].reshape(1,32,32,3)))]
        rea_label = LABELS[int(y_test[idx])]
        plt.title(f'Real: {rea_label}, Predicted: {pre_label}')

    plt.show()


def main():
    # load the data
    x_train, y_train, x_test, y_test = load_data_()

    # build the model
    model = build_model(x_train[0].shape, len(set(y_train.flatten())))
    # or load model
    # model = load_model(r'Assets\cifar10_model.h5')

    # train the model
    r = model.fit(x_train, y_train, epochs=15, verbose=1, validation_data=(x_test, y_test))

    # evaluate the model
    plot_loss(r)
    plot_acc(r)
    
    # confusion matrix
    p_test = model.predict(x_test).argmax(axis=1)
    cm = confusion_matrix(y_test, p_test)
    plot_confusion_matrix(cm, list(range(10)))

    # make predictions
    plot_predictions(x_test, y_test, model)

    # save model
    model.save(r'Assets\cifar10_mnist_model.h5')


if __name__ == '__main__':
    main()
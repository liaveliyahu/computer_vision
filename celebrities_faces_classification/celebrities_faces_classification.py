import os
import cv2
import itertools
import numpy as np
from PIL import Image
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D, Dropout

# Functions
def create_translator(path):
    labels = {}
    i = 0
    for label in os.listdir(path):
        labels[i] = label
        i += 1
    return labels

def get_num_of_images(path):
    num_of_images = 0
    for label in os.listdir(path):
        num_of_images += len(os.listdir(path + os.sep + label))

    return num_of_images

def get_image_shape(path):
    temp_path = path + os.sep + os.listdir(path)[0]
    temp_path += os.sep + os.listdir(temp_path)[0]

    image = cv2.imread(temp_path)

    return image.shape

def load_data(data_path):
    N = get_num_of_images(data_path)
    X_shape = tuple([N] + list(get_image_shape(data_path)))
    
    X = np.empty(X_shape)
    Y = np.empty((N,1))
    
    i, j = 0, 0
    for label in os.listdir(data_path):
        for sample in os.listdir(data_path + os.sep  + label):
            imagePath = data_path + os.sep  + label + os.sep  + sample
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            X[j] = image
            Y[j] = i
            j += 1
        i += 1

    indices = list(range(N))
    shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    X_train = X[:int(N*0.75)]
    Y_train = Y[:int(N*0.75)]
    X_test = X[int(N*0.75):]
    Y_test = Y[int(N*0.75):]

    print('Done loading data!')

    return X_train, Y_train, X_test, Y_test

def build_model(n_outputs):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
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

def plot_predictions(x_test, y_test, model, labels):
    fig = plt.figure(figsize=(10, 10)) 

    for i in range(16):
        fig.add_subplot(4, 4, i+1) 
        idx = np.random.choice(x_test.shape[0])
        int_image = x_test[idx].astype(np.uint8)
        plt.imshow(int_image) 
        plt.axis('off')
        pre_label = labels[np.argmax(model.predict(x_test[idx].reshape(1,100,100,3)))]
        rea_label = labels[int(y_test[idx])]
        plt.title(f'True: {rea_label}\nPredicted: {pre_label}', backgroundcolor='white')

    plt.show()

# Main Fucntion
def main():
    data_path = os.getcwd() + os.sep + 'Data'
    labels =  create_translator(data_path)

    # load the data
    x_train, y_train, x_test, y_test = load_data(data_path)

    # build the model
    ## model = build_model(len(set(y_train.flatten())))
    # or load model
    model = load_model(r'Assets\model.h5')
    
    # train the model
    r = model.fit(x_train, y_train, epochs=5, batch_size=100, verbose=1, validation_data=(x_test, y_test))

    # evaluate the model
    plot_loss(r)
    plot_acc(r)
    
    # confusion matrix
    p_test = model.predict(x_test).argmax(axis=1)
    cm = confusion_matrix(y_test, p_test)
    plot_confusion_matrix(cm, list(range(len(labels))))

    # make predictions
    plot_predictions(x_test, y_test, model, labels)

    # save model
    print('Saving model...')
    model.save(r'Assets\model.h5')
    print('Model saved!')


if __name__ == '__main__':
    main()

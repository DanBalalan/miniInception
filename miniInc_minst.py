import keras
from keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, concatenate
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1))[0:2000]
y_train = to_categorical(y_train, 10)[0:2000]

def inception_Module(inputs):
    tower_one = MaxPooling2D((3,3), strides=(1,1), padding='same')(inputs)
    tower_one = Conv2D(6, (1,1), activation='relu', border_mode='same')(tower_one)

    tower_two = Conv2D(6, (1,1), activation='relu', border_mode='same')(inputs)
    tower_two = Conv2D(6, (3,3), activation='relu', border_mode='same')(tower_two)

    tower_three = Conv2D(6, (1,1), activation='relu', border_mode='same')(inputs)
    tower_three = Conv2D(6, (5,5), activation='relu', border_mode='same')(tower_three)
    x = concatenate([tower_one, tower_two, tower_three], axis=3)
    return x

def Inception_Model(x_train):

    inputs = Input(x_train.shape[1:])

    x = Conv2D(20, (3,3), activation='relu', border_mode='same')(inputs)
    x = inception_Module(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)
    x = inception_Module(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(input=inputs, output=predictions)

    model.compile(loss=keras.losses.categorical_crossentropy,
                 optimizer=keras.optimizers.SGD(lr=0.0001),
                 metrics=['accuracy'])
    return model

model = Inception_Model(x_train)
model.fit(x_train, y_train, epochs=25, shuffle=True,  validation_split=0.1)

import matplotlib.pyplot as plt
def show_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'test_accuracy'], loc='best')
    plt.show()

show_history(model.history)
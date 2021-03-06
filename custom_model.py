from tensorflow import keras
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, BatchNormalization, Flatten, Dropout, Dense

def createModel():
    input_shape=(22, 6726)
    model = Sequential()
    #model.add(Flatten(data_format='channels_first', input_shape=input_shape))
    #D1
    model.add(Dense(6726, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    
    model.add(Dropout(0.5))
    #D2
    model.add(Dense(1680, activation='relu'))
    model.add(BatchNormalization())
    
    #D3
    model.add(Dense(420, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    #model.add(Dropout(0.5))
    #model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    opt_adam = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=opt_adam, metrics=['accuracy'])
    
    return model

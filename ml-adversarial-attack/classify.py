import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.callbacks import ModelCheckpoint

def create_model(input_dim, output_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,))) 
    model.add(Dense(128, activation='relu'))  
    model.add(Dense(64, activation='relu'))  
    model.add(Dense(output_dim, activation='softmax'))  
    return model

input_dim = 784 
output_dim = 10  
learning_rate = 0.001  
batch_size = 32  
epochs = 10 

print()
print("Creating Model...")
print()
model = create_model(input_dim, output_dim)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.save('untrained_ANN_Model.keras')

checkpoint = ModelCheckpoint("untrained_ANN_Model.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

X_train = np.load("train data/X_train.npy", allow_pickle=True)
y_train = np.load("train data/y_train.npy", allow_pickle=True)

print()
print("Training Model...")
print()
y_train = y_train.astype(int)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])

print("Training history:")
print(history.history)

model.save('trained_ANN_Model.keras')

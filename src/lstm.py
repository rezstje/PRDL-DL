import h5py
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from collections import defaultdict
from tensorflow.keras.optimizers import Adam, AdamW
import matplotlib.pyplot as plt
from data_processing import Data_retrieval_and_processing 
from sklearn.utils import shuffle
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv1D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report



train_data_intra = Data_retrieval_and_processing("Final Project data/Intra/train/")
test_data_intra = Data_retrieval_and_processing("Final Project data/Intra/test/")

train_data_cross = Data_retrieval_and_processing("Final Project data/Cross/train/")
test_data_cross1 = Data_retrieval_and_processing("Final Project data/Cross/test1/")
test_data_cross2 = Data_retrieval_and_processing("Final Project data/Cross/test2/")
test_data_cross3 = Data_retrieval_and_processing("Final Project data/Cross/test3/")

x_train_intra = np.asarray(train_data_intra.processed_matrices)
y_train_intra = train_data_intra.one_hot_encoded_labels
x_test_intra = np.asarray(test_data_intra.processed_matrices)
y_test_intra = test_data_intra.one_hot_encoded_labels

x_train_cross = np.asarray(train_data_cross.processed_matrices)
y_train_cross = train_data_cross.one_hot_encoded_labels
x_test_cross1 = np.asarray(test_data_cross1.processed_matrices)
y_test_cross1 = test_data_cross1.one_hot_encoded_labels
x_test_cross2 = np.asarray(test_data_cross2.processed_matrices)
y_test_cross2 = test_data_cross2.one_hot_encoded_labels
x_test_cross3 = np.asarray(test_data_cross3.processed_matrices)
y_test_cross3 = test_data_cross3.one_hot_encoded_labels

x_test_cross = np.concatenate((x_test_cross1, x_test_cross2, x_test_cross3), axis=1)
y_test_cross = y_test_cross1 + y_test_cross2 + y_test_cross3

del train_data_intra, test_data_intra, train_data_cross, test_data_cross1, test_data_cross2, test_data_cross3

x_train_intra, y_train_intra = shuffle(x_train_intra, y_train_intra, random_state = 42)
x_test_intra, y_test_intra = shuffle(x_test_intra, y_test_intra, random_state = 42)

x_train_cross, y_train_cross = shuffle(x_train_cross, y_train_cross, random_state = 42)
x_test_cross, y_test_cross = shuffle(x_test_cross, y_test_cross, random_state = 42)

def make_lstm():
    model = keras.Sequential([
        layers.LSTM(1000, batch_input_shape = x_train_intra.shape, return_sequences = True),
        layers.LSTM(500, batch_input_shape = x_train_intra.shape, return_sequences = True),
        layers.LSTM(250, batch_input_shape = x_train_intra.shape, return_sequences = True),
        layers.LSTM(125, batch_input_shape = x_train_intra.shape, return_sequences = True),
        layers.LSTM(75, batch_input_shape = x_train_intra.shape),
        layers.Dense(y_train_intra.shape[1], activation = "softmax")]
    )

    print(model.summary())

    model.compile(loss = "categorical_crossentropy", optimizer = Adam(learning_rate = 5e-7), metrics = ['accuracy'])
    return model
# hist = model.fit(x_train, y_train, epochs = 250, validation_split = 0.1, callbacks = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 5))
# hist = model.fit(x_train, y_train, epochs = 1000, validation_split = (x_validation, y_validation)) # change to leave-one-out?
# hist = model.fit(x_train, y_train, epochs = 1000)
    
print("INTRA LSTM MODEL bELOW:")
model_lstm_intra = make_lstm()
hist = model_lstm_intra.fit(x_train_intra, y_train_intra, epochs = 500, validation_split = 0.1)
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
_, acc = model_lstm_intra.evaluate(x_test_intra, y_test_intra)
print(acc)

del model_lstm_intra

print("CROSS LSTM MODEL below:")
model_lstm_cross = make_lstm()
hist = model_lstm_cross.fit(x_train_cross, y_train_cross, epochs = 500, validation_split = 0.1)
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
_, acc2 = model_lstm_cross.evaluate(x_test_cross, y_test_cross)
print(acc2)

del model_lstm_cross


def data_generator(data, labels, batch_size):
    num_samples = len(data)
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield data[batch_indices], labels[batch_indices]

def create_1d_cnn(xtrain):
    ### 1d_cnn
    input_shape = [xtrain[0].shape[0],xtrain[0].shape[1]]
    model_1d_cnn_cross = Sequential()

    # Convolutional layers
    model_1d_cnn_cross.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    #model.add(MaxPooling1D(pool_size=2))
    model_1d_cnn_cross.add(Conv1D(64, kernel_size=3, activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    model_1d_cnn_cross.add(Conv1D(128, kernel_size=3, activation='relu'))

    # Flatten layer
    model_1d_cnn_cross.add(Flatten())

    # Fully connected layers
    model_1d_cnn_cross.add(Dense(256, activation='relu'))
    model_1d_cnn_cross.add(Dropout(0.3))
    model_1d_cnn_cross.add(Dense(128, activation='relu'))
    model_1d_cnn_cross.add(Dropout(0.2))

    # Output layer
    num_classes = 4  # Assuming 4 categories
    model_1d_cnn_cross.add(Dense(num_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

    # Compile the model
    model_1d_cnn_cross.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    batch_size = 3

    # Create data generators
    array_3d_train_cross = np.stack(x_train_cross)

    # train the model using some random labels
    train_generator_cross = data_generator(array_3d_train_cross, y_train_cross, batch_size)

    # Create an EarlyStopping callback
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=True)

    # Create a ModelCheckpoint callback
    model_checkpoint = ModelCheckpoint(filepath='..//model//best_model.keras', monitor='loss', save_best_only=True)

    # Train the model using the fit_generator method
    history = model_1d_cnn_cross.fit(train_generator_cross, steps_per_epoch=len(array_3d_train_cross)//2, epochs=500, callbacks=[early_stopping,model_checkpoint])

    return model_1d_cnn_cross, history

model_1d_cnn_cross, _ = create_1d_cnn(x_train_cross)
model_1d_cnn_intra, _ = create_1d_cnn(x_train_intra)
"""
# Plot training loss and accuracy
plt.figure(figsize=(12, 4))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
"""


"""
# Display the model summary
predictions = model_lstm.predict(x_test_cross1)
label_binarizer = LabelBinarizer()
predicted_label = label_binarizer.inverse_transform(predictions)
accuracy = accuracy_score(y_test_cross, predicted_label)
print(f'Accuracy: {accuracy:.2f}')

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test_cross, predicted_label)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Display a classification report
class_report = classification_report(y_test_cross1, predicted_label)
print("Classification Report:\n", class_report)
"""
_, acc1 = model_1d_cnn_cross.evaluate(x_test_cross1, y_test_cross1)
_, acc2 = model_1d_cnn_cross.evaluate(x_test_cross2, y_test_cross2)
_, acc3 = model_1d_cnn_cross.evaluate(x_test_cross3, y_test_cross3)
avg_accuracy = (acc1 + acc2 + acc3) / 3
print(f"accuracy 1D_CNN cross: {avg_accuracy}")

_, acc = model_1d_cnn_intra.evaluate(x_test_intra, y_test_intra)
print(f"accuracy LSTM intra: {acc}")


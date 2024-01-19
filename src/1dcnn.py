import h5py
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Check if GPU is available and visible to TensorFlow
if tf.config.list_physical_devices('GPU'):
    print("GPU is available.")
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
else:
    print("GPU is not available. Switching to CPU.")


useZScore = False
cross = True

## change this to the directory for the train and test set of the intra map
directory_path_train = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/Intra/train")
directory_path_test = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/Intra/test")

if cross:
    directory_path_train = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/Cross/train")
    directory_path_test = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/Cross/test1")

def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name

labels_train = []
all_matrices_train = []
for filename in os.listdir(directory_path_train):
    if os.path.isfile(os.path.join(directory_path_train, filename)):
       file_path = directory_path_train + "//" + filename
    with h5py.File(file_path, 'r') as f:
        dataset_name = get_dataset_name(file_path)
        matrix = f.get(dataset_name)[()]
        all_matrices_train.append(matrix)
        labels_train.append(filename)

labels_test = []
all_matrices_test = []
for filename in os.listdir(directory_path_test):
    if os.path.isfile(os.path.join(directory_path_test, filename)):
       file_path = directory_path_test + "//" + filename
    with h5py.File(file_path, 'r') as f:
        dataset_name = get_dataset_name(file_path)
        matrix = f.get(dataset_name)[()]
        all_matrices_test.append(matrix)
        labels_test.append(filename)

## Extract the main label of the filename for train
all_labels_train = []

for s in labels_train:
    if "rest" in s:
        all_labels_train.append("Resting task")
    elif "task_motor" in s:
        all_labels_train.append("Motor task")
    elif "task_story" in s:
        all_labels_train.append("Math and story task")
    elif "task_working_memory" in s:
        all_labels_train.append("Working memory task")

## Extract the main label of the filename for test
all_labels_test = []
for s in labels_test:
    if "rest" in s:
        all_labels_test.append("Resting task")
    elif "task_motor" in s:
        all_labels_test.append("Motor task")
    elif "task_story" in s:
        all_labels_test.append("Math and story task")
    elif "task_working_memory" in s:
        all_labels_test.append("Working memory task")

# Using a minmax scaler
scaler = MinMaxScaler()
scaled_arrays_train = []
for array in all_matrices_train:
    array_reshaped = array.reshape(-1, 1)
    scaled_array_reshaped = scaler.fit_transform(array_reshaped)
    scaled_array = scaled_array_reshaped.reshape(array.shape)
    scaled_arrays_train.append(scaled_array)

# Using a minmax scaler
scaler = MinMaxScaler()
scaled_arrays_test = []
for array in all_matrices_test:
    array_reshaped = array.reshape(-1, 1)
    scaled_array_reshaped = scaler.fit_transform(array_reshaped)
    scaled_array = scaled_array_reshaped.reshape(array.shape)
    scaled_arrays_test.append(scaled_array)

# Z-score Normalization
zscore_scaler = StandardScaler()

normalized_arrays_train = []
for array in all_matrices_train:
    array_reshaped = array.reshape(-1, 1)
    normalized_array_reshaped = zscore_scaler.fit_transform(array_reshaped)
    normalized_array = normalized_array_reshaped.reshape(array.shape)
    normalized_arrays_train.append(normalized_array)

normalized_arrays_test = []
for array in all_matrices_test:
    array_reshaped = array.reshape(-1, 1)
    normalized_array_reshaped = zscore_scaler.fit_transform(array_reshaped)
    normalized_array = normalized_array_reshaped.reshape(array.shape)
    normalized_arrays_test.append(normalized_array)

if useZScore:
    scaled_arrays_train = normalized_arrays_train
    scaled_arrays_test = normalized_arrays_test

# Downsampling
## taking the mean of an interval of 8 frames.
all_matrices_downsampled_train = []
for i in range(0,len(scaled_arrays_train)):
    downsampling_factor = 8
    current_matrix = scaled_arrays_train[i]
    downsampled_columns = current_matrix.shape[1] // downsampling_factor
    reshaped_array = current_matrix[:, :downsampled_columns * downsampling_factor].reshape(current_matrix.shape[0], downsampled_columns, downsampling_factor)
    downsampled_array = np.mean(reshaped_array,axis = 2)
    all_matrices_downsampled_train.append(downsampled_array)

## taking the mean of an interval of 8 frames.
all_matrices_downsampled_test = []
for i in range(0,len(scaled_arrays_test)):
    downsampling_factor = 8
    current_matrix = scaled_arrays_test[i]
    downsampled_columns = current_matrix.shape[1] // downsampling_factor
    reshaped_array = current_matrix[:, :downsampled_columns * downsampling_factor].reshape(current_matrix.shape[0], downsampled_columns, downsampling_factor)
    downsampled_array = np.mean(reshaped_array,axis = 2)
    all_matrices_downsampled_test.append(downsampled_array)

# this gives the label rest the value 2, the label motor task the value 1,
# the label math and story the value 0 and the working memory task the value 3
desired_classes_order = ["Resting task", "Motor task", "Math and story task", "Working memory task"]
label_binarizer = LabelBinarizer()

one_hot_encoded = label_binarizer.fit_transform(all_labels_train)

# memory clear
all_matrices_train = []
all_matrices_test = []
current_matrix = []
scaled_arrays_train = []
scaled_arrays_test = []
normalized_arrays_train = []
normalized_arrays_test = []
scaled_array_reshaped = []
reshaped_array = []

input_shape = [all_matrices_downsampled_train[0].shape[0],all_matrices_downsampled_train[0].shape[1]]
model = Sequential()

# Convolutional layers
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(Conv1D(128, kernel_size=3, activation='relu'))

# Flatten layer
model.add(Flatten())

# Fully connected layers
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

# Output layer
num_classes = 4  # Assuming 4 categories
model.add(Dense(num_classes, activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

def data_generator(data, labels, batch_size):
    num_samples = len(data)
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield data[batch_indices], labels[batch_indices]

batch_size = 3

# Create data generators
array_3d_train = np.stack(all_matrices_downsampled_train)
array_3d_test = np.stack(all_matrices_downsampled_test)
from sklearn.model_selection import train_test_split

array_3d_train, array_3d_val, labels_train, labels_val = train_test_split(array_3d_train, one_hot_encoded, test_size=0.1, random_state=42)

# train the model using some random labels
train_generator = data_generator(array_3d_train, labels_train, batch_size)
val_generator = data_generator(array_3d_val, labels_val, batch_size)

# Train the model using the fit_generator method
history = model.fit(train_generator,
                    steps_per_epoch=len(array_3d_train),
                    epochs=50,
                    validation_data=val_generator,
                    validation_steps=len(array_3d_val),
                    )


# Plot Training loss on validation and train
sns.set_theme(style = "whitegrid")
plt.figure(figsize = (8,6))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title('1D CNN: Training and Validation Loss over Epochs for Intra Data')
plt.legend(frameon = False)
sns.despine()
plt.savefig('1dcnn_intra', transparent = True)
plt.show()

# Plot training and validation accuracy
plt.figure(figsize = (8,6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over Epochs')
plt.legend()
plt.show()

# Display the model summary
predictions = model.predict(array_3d_test)
predicted_label = label_binarizer.inverse_transform(predictions)
accuracy = accuracy_score(all_labels_test, predicted_label)
print(f'Accuracy: {accuracy:.2f}')

# Create a confusion matrix
conf_matrix = confusion_matrix(all_labels_test, predicted_label)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Display a classification report
class_report = classification_report(all_labels_test, predicted_label)
print("Classification Report:\n", class_report)

# Save the entire model as a `.keras` zip archive.
model.save('1dcnn.keras')

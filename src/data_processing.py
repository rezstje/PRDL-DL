import h5py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer


class Data_retrieval_and_processing:

    def __init__(self, path):
        self.directory_path = path

        self.raw_labels, self.raw_matrices = self.get_labels_and_matrices()
        self.down_sampling_factor = 8
        self.process_data()

    def get_dataset_name(self, file_name_with_dir):
        filename_without_dir = file_name_with_dir.split('/')[-1]
        temp = filename_without_dir.split('_')[:-1]
        dataset_name = "_".join(temp)
        return dataset_name
    
    def get_labels_and_matrices(self):
        labels = []
        matrices = []
        for filename in os.listdir(self.directory_path):
            if os.path.isfile(os.path.join(self.directory_path, filename)):
                file_path = self.directory_path + "//" + filename
            with h5py.File(file_path, 'r') as f:
                dataset_name = self.get_dataset_name(file_path)
                matrix = f.get(dataset_name)[()]
                matrices.append(matrix)
                labels.append(filename)
        return labels, matrices

    def min_max_all_matrices(self, all_matrices):
        #using a minmax scaler
        scaler = MinMaxScaler()
        scaled_arrays = []
        for array in all_matrices:
            array_reshaped = array.reshape(-1, 1)
            scaled_array_reshaped = scaler.fit_transform(array_reshaped)
            scaled_array = scaled_array_reshaped.reshape(array.shape)
            scaled_arrays.append(scaled_array)
        return scaled_arrays

    def downsampling_matrices(self, all_matrices, factor):
        all_matrices_downsampled = []
        for i in range(0,len(all_matrices)):
            current_matrix = all_matrices[i]
            downsampled_columns = current_matrix.shape[1] // factor
            reshaped_array = current_matrix[:, :downsampled_columns * factor].reshape(current_matrix.shape[0], downsampled_columns, factor)
            downsampled_array = np.mean(reshaped_array,axis = 2)
            all_matrices_downsampled.append(downsampled_array)
        return all_matrices_downsampled
    

    def process_data(self):
        scaled_arrays = self.min_max_all_matrices(self.raw_matrices)
        self.processed_matrices = self.downsampling_matrices(scaled_arrays, self.down_sampling_factor)

        all_labels = []
        #labels
        for s in self.raw_labels:
            if "rest" in s:
                all_labels.append("Resting task")
            elif "task_motor" in s:
                all_labels.append("Motor task")
            elif "task_story" in s:
                all_labels.append("Math and story task")
            elif "task_working_memory" in s:
                all_labels.append("Working memory task")

        # this gives the label rest the value 2, the label motor task the value 1,
        # the label math and story the value 0 and the working memory task the value 3
        #desired_classes_order = ["Resting task", "Motor task", "Math and story task", "Working memory task"]
        label_binarizer = LabelBinarizer()
        self.one_hot_encoded_labels = label_binarizer.fit_transform(all_labels)

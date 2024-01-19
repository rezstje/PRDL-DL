import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

real = [[0, 0, 1, 0],
 [0, 0, 1, 0],
 [0, 0, 1, 0],
 [0, 0, 1, 0],
 [0, 1, 0, 0],
 [0, 1, 0, 0],
 [0, 1, 0, 0],
 [0, 1, 0, 0],
 [1, 0, 0, 0],
 [1, 0, 0, 0],
 [1, 0, 0, 0],
 [1, 0, 0, 0],
 [0, 0, 0, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 1],
 [0, 0, 1, 0],
 [0, 0, 1, 0],
 [0, 0, 1, 0],
 [0, 0, 1, 0],
 [0, 1, 0, 0],
 [0, 1, 0, 0],
 [0, 1, 0, 0],
 [0, 1, 0, 0],
 [1, 0, 0, 0],
 [1, 0, 0, 0],
 [1, 0, 0, 0],
 [1, 0, 0, 0],
 [0, 0, 0, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 1],
 [0, 0, 1, 0],
 [0, 0, 1, 0],
 [0, 0, 1, 0],
 [0, 0, 1, 0],
 [0, 1, 0, 0],
 [0, 1, 0, 0],
 [0, 1, 0, 0],
 [0, 1, 0, 0],
 [1, 0, 0, 0],
 [1, 0, 0, 0],
 [1, 0, 0, 0],
 [1, 0, 0, 0],
 [0, 0, 0, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 1]]

predicted = [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]

real_1 = np.array(real)
desired_classes_order = ["Resting task", "Motor task", "Math and story task", "Working memory task"]
real_2 = []
for i in range(0,len(real_1)):
 if np.array_equal(real_1[i],[0,0,1,0]):
  real_2.append("Resting")
  print(i)
 elif np.array_equal(real_1[i],[0,1,0,0]):
  real_2.append("Motor")
 elif np.array_equal(real_1[i],[1,0,0,0]):
  real_2.append("Math & story")
 elif np.array_equal(real_1[i],[0,0,0,1]):
  real_2.append("Working memory")

pred_2 = []
for i in range(0,len(predicted)):
 if np.array_equal(predicted[i],[0,0,1,0]):
  pred_2.append("Resting")
  print(i)
 elif np.array_equal(predicted[i],[0,1,0,0]):
  pred_2.append("Motor")
 elif np.array_equal(predicted[i],[1,0,0,0]):
  pred_2.append("Math & story")
 elif np.array_equal(predicted[i],[0,0,0,1]):
  pred_2.append("Working memory")

cm = confusion_matrix(real_2, pred_2)

# Get class labels
classes = unique_labels(real_2, pred_2)

# Create a seaborn heatmap for visualization
# Set the text size
label_size = 16

# Create a seaborn heatmap for visualization
plt.figure(figsize=(len(classes)+6, len(classes)+4))
heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)

# Set the text size for x-axis and y-axis labels
plt.xticks(fontsize=label_size)
plt.yticks(fontsize=label_size)
plt.xlabel('Predicted Labels', fontsize=label_size)
plt.ylabel('Actual Labels', fontsize=label_size)
plt.title('Confusion Matrix 1D CNN Cross Data', fontsize=label_size)
plt.savefig("1DCNN_real")
plt.show()



predicted = ("Working memory task",
"Math and story task",
"Motor task",
"Working memory task",
"Resting task",
"Working memory task",
"Resting task",
"Resting task",
"Resting task",
"Resting task",
"Math and story task",
"Resting task",
"Working memory task",
"Resting task",
"Motor task",
"Working memory task",
"Working memory task",
"Working memory task",
"Resting task",
"Working memory task",
"Motor task",
"Working memory task",
"Working memory task",
"Resting task",
"Resting task",
"Working memory task",
"Working memory task",
"Working memory task",
"Resting task",
"Resting task",
"Motor task",
"Motor task",
"Working memory task",
"Resting task",
"Working memory task",
"Math and story task",
"Working memory task",
"Resting task",
"Working memory task",
"Working memory task",
"Resting task",
"Motor task",
"Working memory task",
"Resting task",
"Motor task",
"Working memory task",
"Working memory task",
"Working memory task")

pred = np.array(predicted)

real = ("Working memory task",
"Math and story task",
"Motor task",
"Motor task",
"Resting task",
"Motor task",
"Math and story task",
"Resting task",
"Math and story task",
"Resting task",
"Math and story task",
"Working memory task",
"Motor task",
"Resting task",
"Working memory task",
"Working memory task",
"Math and story task",
"Math and story task",
"Working memory task",
"Math and story task",
"Working memory task",
"Motor task",
"Working memory task",
"Resting task",
"Resting task",
"Working memory task",
"Motor task",
"Motor task",
"Resting task",
"Resting task",
"Motor task",
"Math and story task",
"Math and story task",
"Resting task",
"Motor task",
"Motor task",
"Math and story task",
"Resting task",
"Working memory task",
"Working memory task",
"Resting task",
"Math and story task",
"Math and story task",
"Resting task",
"Motor task",
"Working memory task",
"Motor task",
"Working memory task")

real_1 = np.array(real)
desired_classes_order = ["Resting task", "Motor task", "Math and story task", "Working memory task"]
real_2 = []
for i in range(0,len(real_1)):
 if np.array_equal(real_1[i],"Resting task"):
  real_2.append("Resting")
  print(i)
 elif np.array_equal(real_1[i],"Motor task"):
  real_2.append("Motor")
 elif np.array_equal(real_1[i],"Math and story task"):
  real_2.append("Math & story")
 elif np.array_equal(real_1[i],"Working memory task"):
  real_2.append("Working memory")

pred_2 = []
for i in range(0,len(predicted)):
 if np.array_equal(predicted[i],"Resting task"):
  pred_2.append("Resting")
  print(i)
 elif np.array_equal(predicted[i],"Motor task"):
  pred_2.append("Motor")
 elif np.array_equal(predicted[i],"Math and story task"):
  pred_2.append("Math & story")
 elif np.array_equal(predicted[i],"Working memory task"):
  pred_2.append("Working memory")

cm = confusion_matrix(real_2, pred_2)

# Get class labels
classes = unique_labels(real_2, pred_2)

# Create a seaborn heatmap for visualization
# Set the text size
label_size = 16

# Create a seaborn heatmap for visualization
plt.figure(figsize=(len(classes)+6, len(classes)+4))
heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)

# Set the text size for x-axis and y-axis labels
plt.xticks(fontsize=label_size)
plt.yticks(fontsize=label_size)
plt.xlabel('Predicted Labels', fontsize=label_size)
plt.ylabel('Actual Labels', fontsize=label_size)
plt.title('Confusion Matrix BiLSTM Cross Data', fontsize=label_size)
plt.savefig("BiLSTM_real")
plt.show()
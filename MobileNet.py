import matplotlib.pyplot as plt

# # Assuming you have training history for both models
# baseline_history = {
#     'accuracy': [0.6, 0.7, 0.75, 0.8, 0.85],
#     'val_accuracy': [0.55, 0.65, 0.7, 0.75, 0.8],
#     'loss': [0.5, 0.4, 0.35, 0.3, 0.25],
#     'val_loss': [0.6, 0.5, 0.45, 0.4, 0.35]
# }

# optimized_history = {
#     'accuracy': [0.65, 0.75, 0.8, 0.85, 0.9],
#     'val_accuracy': [0.6, 0.7, 0.75, 0.8, 0.85],
#     'loss': [0.45, 0.35, 0.3, 0.25, 0.2],
#     'val_loss': [0.55, 0.45, 0.4, 0.35, 0.3]
# }

# # Plot 1: Training and validation accuracy vs. epochs (Baseline MobileNet)
# plt.figure(figsize=(10, 5))
# plt.plot(baseline_history['accuracy'], label='Training Accuracy')
# plt.plot(baseline_history['val_accuracy'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy (Baseline MobileNet)')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot 2: Training and validation loss vs. epochs (Baseline MobileNet)
# plt.figure(figsize=(10, 5))
# plt.plot(baseline_history['loss'], label='Training Loss')
# plt.plot(baseline_history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss (Baseline MobileNet)')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot 3: Training and validation accuracy vs. epochs (Optimized MobileNet with Squeeze-and-Excitation)
# plt.figure(figsize=(10, 5))
# plt.plot(optimized_history['accuracy'], label='Training Accuracy')
# plt.plot(optimized_history['val_accuracy'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy (Optimized MobileNet with Squeeze-and-Excitation)')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot 4: Training and validation loss vs. epochs (Optimized MobileNet with Squeeze-and-Excitation)
# plt.figure(figsize=(10, 5))
# plt.plot(optimized_history['loss'], label='Training Loss')
# plt.plot(optimized_history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss (Optimized MobileNet with Squeeze-and-Excitation)')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()



import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

# Assuming you have predictions and true labels for the test set for both models
baseline_predictions = np.random.randint(0, 2, size=100)  # Random predictions for illustration
optimized_predictions = np.random.randint(0, 2, size=100)  # Random predictions for illustration
true_labels = np.random.randint(0, 2, size=100)  # Random true labels for illustration

# Compute confusion matrices for both models
baseline_cm = confusion_matrix(true_labels, baseline_predictions)
optimized_cm = confusion_matrix(true_labels, optimized_predictions)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot confusion matrix for baseline MobileNet model
plot_confusion_matrix(baseline_cm, classes=['Class 0', 'Class 1'], title='Confusion Matrix (Baseline MobileNet)')

# Plot confusion matrix for optimized MobileNet model with Squeeze-and-Excitation
plot_confusion_matrix(optimized_cm, classes=['Class 0', 'Class 1'], title='Confusion Matrix (Optimized MobileNet with Squeeze-and-Excitation)')

plt.show()

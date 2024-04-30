import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
import numpy as np
from PIL import Image
from tkinter import Tk, filedialog, Label, Button
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = np.expand_dims(x_train, axis=-1) / 255.0
x_test = np.expand_dims(x_test, axis=-1) / 255.0

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
start_time = time.time()
model.fit(x_train, y_train, epochs=1)
end_time = time.time()

# Calculate accuracy
_, test_accuracy = model.evaluate(x_test, y_test)

# Calculate model size
model_size = model.count_params()

# Calculate average inference speed
# Assuming a batch of images to predict
batch_size = 32
x_batch = x_train[:batch_size]
start_time_inference = time.time()
model.predict(x_batch)
end_time_inference = time.time()
inference_speed = (end_time_inference - start_time_inference) / batch_size

# Generate confusion matrix
y_pred = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Sample predictions
def sample_predictions():
    for i in range(5):
        index = np.random.randint(len(x_test))
        img = x_test[index]
        true_label = y_test[index]
        pred = np.argmax(model.predict(np.expand_dims(img, axis=0)))
        print(f"Sample {i+1}: True Label: {true_label}, Predicted Label: {pred}")

# Print results
print("Specific Achievements and Quantitative Results:")
print("Accuracy:", test_accuracy)
print("Model Size (number of parameters):", model_size)
print("Average Inference Speed (per image):", inference_speed)
print("Training Time:", end_time - start_time)
print("Sample Predictions:")
sample_predictions()

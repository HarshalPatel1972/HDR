import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tkinter import Tk, filedialog, Label, Button
import time

# Load MNIST dataset
(x_train, y_train), (_, _) = mnist.load_data()

# Preprocess data
x_train = np.expand_dims(x_train, axis=-1) / 255.0

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
model.fit(x_train, y_train, epochs= 7)

# Function to predict digit from uploaded image
def predict_digit(image_path):
    start_time = time.time()
    img = Image.open(image_path)
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    digit = np.argmax(pred)
    end_time = time.time()
    return digit, end_time - start_time

def upload_image():
    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                                filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("all files", "*.*")))
    digit, time_taken = predict_digit(root.filename)
    root.destroy()
    result_label.config(text=f"The predicted digit is: {digit}\nTime taken: {time_taken:.2f} seconds")
    # Calculate model accuracy
    _, accuracy = model.evaluate(x_train, y_train, verbose=0)
    accuracy_label.config(text=f"Model Accuracy: {accuracy:.2%}")

# GUI setup
root = Tk()
root.title("Handwritten Digit Recognition")
root.geometry("300x200")

upload_button = Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

result_label = Label(root, text="")
result_label.pack(pady=10)

accuracy_label = Label(root, text="")
accuracy_label.pack()

root.mainloop()

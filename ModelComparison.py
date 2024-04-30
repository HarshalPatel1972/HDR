import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tkinter import Tk, filedialog, Label, Button
import time

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data for CNN model
x_train_cnn = np.expand_dims(x_train, axis=-1) / 255.0
x_test_cnn = np.expand_dims(x_test, axis=-1) / 255.0

# Preprocess data for MobileNet model
x_train_mobilenet = np.expand_dims(x_train, axis=-1)
x_train_mobilenet = np.repeat(x_train_mobilenet, 3, axis=-1)  # Convert grayscale to RGB
x_train_mobilenet_resized = [np.array(Image.fromarray(img).resize((224, 224))) for img in x_train_mobilenet]
x_train_mobilenet = np.array(x_train_mobilenet_resized) / 255.0

# Define and train the basic CNN model
basic_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

basic_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

start_time_basic = time.time()
basic_model.fit(x_train_cnn, y_train, epochs=1)
end_time_basic = time.time()

# Define MobileNet model without top layers
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)
mobilenet_model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile MobileNet model
mobilenet_model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

start_time_mobilenet = time.time()
mobilenet_model.fit(x_train_mobilenet, y_train, epochs=10)
end_time_mobilenet = time.time()

# Evaluate models
_, basic_model_accuracy = basic_model.evaluate(x_test_cnn, y_test)
_, mobilenet_model_accuracy = mobilenet_model.evaluate(x_train_mobilenet, y_train)

# Print results
print("Basic CNN Model:")
print("Accuracy:", basic_model_accuracy)
print("Training Time:", end_time_basic - start_time_basic)
print()

print("MobileNet Model:")
print("Accuracy:", mobilenet_model_accuracy)
print("Training Time:", end_time_mobilenet - start_time_mobilenet)

# Function to predict digit from uploaded image using MobileNet
def predict_digit_mobilenet(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = mobilenet_model.predict(img)
    digit = np.argmax(pred)
    return digit

# Function to predict digit from uploaded image using Basic CNN
def predict_digit_basic(image_path):
    img = Image.open(image_path)
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = basic_model.predict(img)
    digit = np.argmax(pred)
    return digit

def upload_image():
    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                                filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("all files", "*.*")))
    digit_mobilenet = predict_digit_mobilenet(root.filename)
    digit_basic = predict_digit_basic(root.filename)
    root.destroy()
    result_label.config(text=f"MobileNet predicted digit: {digit_mobilenet}\nBasic CNN predicted digit: {digit_basic}")

# GUI setup
root = Tk()
root.title("Handwritten Digit Recognition")
root.geometry("300x150")

upload_button = Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

result_label = Label(root, text="")
result_label.pack()

root.mainloop()

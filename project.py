import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#preprocessing steps: normalizing the Data to fit a 0-1 format instead of 0-255
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(256, activation='relu'))
#rectified linear unit -> for x<=0 -> y = 0, else: y=x
model.add(tf.keras.layers.Dense(256, activation='relu'))


#output_layer: 10 neurons since we have 10 different labels.
model.add(tf.keras.layers.Dense(10, activation='softmax'))

#initializing the Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#training the model using the training data
model.fit(x_train, y_train, epochs = 3)

#model.save('handwritten.model')

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)


# Plot loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(0, loss, 'bo-', label='Training Loss')  # 'bo-' means blue circle markers with lines

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(0, accuracy, 'go-', label='Training Accuracy')  # 'go-' means green circle markers with lines
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.grid(True)

# Show plots
plt.tight_layout()
plt.show()

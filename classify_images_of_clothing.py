import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0 #scale data points to between 0 and 1 to make computations simpler
test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #flatten list of list of pixels to pass data to individual neurons
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax") #all values in this layer add up to 1
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Accuracy on Test Data: ", test_acc)

prediction = model.predict(test_images)

'''
print(prediction[0]) # array of 10 output neurons listing the probabilities that the picture at index 0 is a a certain class
print(class_names[np.argmax(prediction[0])]) # argmax finds the index of the probability that is closest to 1 (the max)
'''

for i in range(5):
    j = random.randint(0, len(test_images)-1)
    plt.grid(False)
    plt.imshow(test_images[j], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[j]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[j])])
    plt.show()

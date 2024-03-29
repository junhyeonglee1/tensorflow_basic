import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEl'] = '2'

tf.random.set_seed(777)  # for reproducibility
## Tensorflow2.0 버전에 맞게 Keras를 활용한 Fahion MNIST 를 분류모델 생성 ##
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
##Fashion MNIST Data 확인 - 4번째 배열 드레스
plt.imshow(train_images[3])
plt.colorbar()
plt.grid(False)
##0~1사이의 값으로 정규화 및 Data 출력
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
#Tensorflow keras API를 통해 모델에 대한 정의
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
#Adam Optimizer과 Cross Entropy Loss 선언
#Epoch로 학습할 Data로 학습 수행
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
#모델을 평가할 Test 데이타에 대한 정의
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
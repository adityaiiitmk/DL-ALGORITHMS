import tensorflow as tf
from numpy import unique, argmax
import numpy as np
import pandas as pd


(train_x, y_train), (test_x, y_test) = tf.keras.datasets.mnist.load_data()


print("--------------------------------------\n")
print("Size of training Data Loaded:\n")
print('Train: X=%s, y=%s' % (train_x.shape, y_train.shape))
print('Test: X=%s, y=%s' % (test_x.shape, y_test.shape))
print("--------------------------------------\n")


# reshaping train and test sets 

train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], 1))
test_x = test_x .reshape((test_x.shape[0], test_x.shape[1], test_x.shape[2], 1))

#printing the shapes 
print("--------------------------------------\n")
print("Size of training Data Reshaped:\n")
print('Train: X=%s, y=%s' % (train_x.shape, y_train.shape))
print('Test: X=%s, y=%s' % (test_x.shape, y_test.shape))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Normalaising the Dataset. \n")

train_x = train_x.astype('float')/255
test_x = test_x.astype('float')/255
shape = train_x.shape[1:]
print("--------------------------------------\n")


print("CNN Model Creation \n")

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape= shape))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Conv2D(48, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2)))

model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
print("--------------------------------------\n")

model.summary()


model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics= ['accuracy'])

print("--------------------------------------\n")
print("Training Started.\n")
history = model.fit(train_x, y_train, epochs=10, batch_size = 128, validation_split = 0.2)
print("Training Finished.\n")
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Model Evalutaion Phase.\n")
loss,accuracy= model.evaluate(test_x, y_test)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")


results = model.predict(test_x)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Predicted Label")
submission = pd.concat([pd.Series(y_test,name = "Actual Label"),results],axis = 1)
submission.to_csv("CNN/results/MNIST-CNN.csv",index=False)

# 7 classes of cancer
# akiec, bcc, bkl, df, mel, nv, vasc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image

np.random.seed(42)
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image

# Importing the dataset
skin_data = pd.read_csv(r'C:\Users\nikhi\archive\HAM10000_metadata.csv')
# size of the image accepted by the model
size = 64

# These are the 7 classes of cancer in the dataset
le = LabelEncoder()
le.fit(skin_data['dx'])
LabelEncoder()
print(list(le.classes_))

 # Giving unique number to each type
skin_data['label'] = le.transform(skin_data['dx'])
print(skin_data.sample(10))

# For data visualization
# type vs count
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
types = skin_data['dx'].unique()
count = skin_data['dx'].value_counts()
ax.bar(types,count)
plt.show()

# sex vs count
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
sex = skin_data['sex'].unique()
count1 = skin_data['sex'].value_counts()
ax.bar(sex,count1)
plt.show()

# area vs count
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
area = skin_data['localization'].unique()
count2 = skin_data['localization'].value_counts()
ax.bar(area,count2)
plt.show()

plt.tight_layout()
plt.show()

# getting the count of each class
print(skin_data['label'].value_counts())

# joining two folders of images
image_path = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(r'C:/Users/nikhi/archive/', '*', '*.jpg'))}


skin_data['path'] = skin_data['image_id'].map(image_path.get)

# resizing all the images to 64*64
skin_data['image'] = skin_data['path'].map(lambda x: np.asarray(Image.open(x).resize((size,size))))

# printing some images of the dataset
n_samples = 5  

fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         skin_data.sort_values(['dx']).groupby('dx')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
        
# converting the image column in dataframe to numpy array
X = np.asarray(skin_data['image'].tolist())
# normalization
X = X/255
Y=skin_data['label']  
Y_cat = to_categorical(Y, num_classes=7)
# dividing the dataset for training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)


# defining the architecture of the model
# we will be using 4 layered CNN
num_classes = 7

model = Sequential()
model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(size,size, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3),activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(32))
model.add(Dense(7, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

# training the model of the training dataset
batch_size = 16 
epochs = 50

history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size = batch_size,
    validation_data=(x_test, y_test),
    verbose=2)

# testing the trained model on testing dataset
score = model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])

# plotting the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# saving the model for future use 
model.save('scp.h5')

# Prediction on test data
y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred, axis = 1) 
# Convert test data to one hot vectors
y_true = np.argmax(y_test, axis = 1) 

#Print confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

fig, ax = plt.subplots(figsize=(6,6))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)

#PLot fractional incorrect misclassifications
incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
plt.bar(np.arange(7), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')

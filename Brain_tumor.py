import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,matthews_corrcoef,classification_report,confusion_matrix,ConfusionMatrixDisplay,f1_score
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
import cv2
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import ipywidgets as widgets
import io
from PIL import Image
from IPython.display import display,clear_output
from warnings import filterwarnings
import seaborn as sns
gpu_device_name = tf.test.gpu_device_name()

if gpu_device_name:
    print('GPU kullanılıyor: {}'.format(gpu_device_name))
else:
    print("GPU bulunamadı. TensorFlow yalnızca CPU üzerinde çalışıyor.")

###################################################################################################################
#################################Brain Tumor Classification MRI###################################################
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']

X_train = []
y_train = []
image_size = 224
for i in labels:
    folderPath = os.path.join('Bitirme_Projesi/Datasets/brain-tumor-classification-mri', 'Training', i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        y_train.append(i)

for i in labels:
    folderPath = os.path.join('Bitirme_Projesi/Datasets/brain-tumor-classification-mri', 'Testing', i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        y_train.append(i)

X = np.array(X_train)
y = np.array(y_train)

X.shape,y.shape


X, y = shuffle(X,y, random_state=101)

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.1,random_state=101)

"""from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Label encoding
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# One-hot encoding
y_train_cat = to_categorical(y_train_encoded)
y_test_cat = to_categorical(y_test_encoded)"""
#encode labels
y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train = np.array(y_train_new)
y_train_cat = tf.keras.utils.to_categorical(y_train)


y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test = np.array(y_test_new)
y_test_cat = tf.keras.utils.to_categorical(y_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.losses import CategoricalCrossentropy

model  = Sequential()
model.add(Conv2D(64,(3,3),activation='relu',input_shape=(image_size,image_size,3)))
model.add(MaxPooling2D((2,2)),)

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


history = model.fit(X_train,y_train,validation_split=0.2, epochs=10, batch_size=16)
model.save('model.h5')



from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


from tensorflow.keras.models import load_model
model=load_model('model.h5')

classification_report_print(X_test,y_test)


from sklearn.metrics import classification_report

def classification_report_print(X_test, y_test):
    y_pred = model.predict(X_test)  # model burada tanımlı olmalı
    y_classes = np.argmax(y_pred, axis=1)
    print('classification Report \n', classification_report(y_test, y_classes))


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)



# Confusion matrix oluştur
cm = confusion_matrix(y_test, y_pred_classes)

# Görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']


img = cv2.imread(f'C:/Users/yekta/Downloads/meningioma_parasagittal_high.jpg')
img = cv2.resize(img,(224,224))
img_array = np.array(img)
img_array = img_array.reshape(1,224,224,3)
a=model.predict(img_array)
labels[a.argmax()]



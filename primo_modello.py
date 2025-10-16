#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.layers as layer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# In[2]:


train_augmentation = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=20,
    zoom_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.1,   
    height_shift_range=0.1 )
train_gen = train_augmentation.flow_from_directory(
        "/home/ocelot/Desktop/CNN_RPS/Archive/train",
        target_size=(100,150),
        batch_size=32,
        class_mode="categorical",
        color_mode="grayscale",
        seed=42

)

val_gen = ImageDataGenerator(rescale = 1./255).flow_from_directory(
    "/home/ocelot/Desktop/CNN_RPS/Archive/validation",
     target_size=(100,150),
     batch_size=32,
     class_mode="categorical",
     color_mode="grayscale",
     seed=42)


# In[3]:


early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
model = tensorflow.keras.models.Sequential([
    layer.InputLayer(shape=(100,150,1)),
    layer.Conv2D(filters=(32), kernel_size=(3,3), activation="relu"),
    layer.MaxPooling2D(pool_size=(2,2)),
    layer.Flatten(),
    layer.Dense(64, activation="relu"),
    layer.Dense(3, activation="softmax")
])


# In[4]:


get_ipython().run_cell_magic('time', '', 'model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])\nhistory=model.fit(train_gen, epochs=100, callbacks=[early_stop], validation_data=val_gen)\n')


# In[5]:


test_gen = ImageDataGenerator(rescale = 1./255).flow_from_directory(
    "/home/ocelot/Desktop/CNN_RPS/Archive/test",
     target_size=(100,150),
     batch_size=32,
     class_mode="categorical",
     color_mode="grayscale",
     shuffle=False,
     seed=42)


# In[6]:


fig, ax = plt.subplots(1,2)
fig.set_size_inches(10,5)
ax[0].plot(history.history["loss"], label="training loss")
ax[0].plot(history.history["val_loss"], label="validation loss")
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("loss")
ax[0].legend()
ax[1].plot(history.history["accuracy"], label="training accuracy")
ax[1].plot(history.history["val_accuracy"], label="validation accuracy")
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("accuracy")
ax[1].legend()


# In[7]:


testLoss, testAccuracy = model.evaluate(test_gen)


# In[8]:


import numpy as np
from sklearn.metrics import classification_report
pred = model.predict(test_gen)
pred = np.argmax(pred, axis=1)
real_label = test_gen.classes
keys = test_gen.class_indices.keys()
keys = list(keys)
report = classification_report(real_label, pred, target_names=keys, output_dict=True)


# In[9]:


report["scissors"]


# In[10]:


report["paper"]


# In[11]:


report["rock"]


# In[12]:


import cv2
errori_indice=[]
for i in range(len(real_label)):
    if real_label[i] != pred[i]:
        errori_indice.append(i)
file = test_gen.filepaths
errori = {"immagini":[], "LabelGiusta":[], "LabelPred":[]}
for i in errori_indice:
    plt.figure(figsize=(4, 4))
    plt.imshow(cv2.cvtColor(cv2.imread(file[i]), cv2.COLOR_BGR2RGB))
    plt.title("Label predetta: {} Label vera: {}".format(keys[pred[i]], keys[real_label[i]]))


# In[13]:


model.save("primo.keras")


# In[ ]:





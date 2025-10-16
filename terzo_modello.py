#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow 
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.layers as layer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf


# In[4]:


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




# In[5]:


early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
model = tensorflow.keras.models.Sequential([
    layer.InputLayer(shape=(100,150,1)),
    layer.Conv2D(32, (3,3), activation='relu', padding='same'),
    layer.Conv2D(32, (3,3), activation='relu', padding='same'),
    layer.MaxPooling2D((2,2)),
    layer.Dropout(0.25),
    layer.Conv2D(64, (3,3), activation='relu', padding='same'),
    layer.Conv2D(64, (3,3), activation='relu', padding='same'),
    layer.MaxPooling2D((2,2)),
    layer.Dropout(0.25),
    layer.Conv2D(128, (3,3), activation='relu', padding='same'),
    layer.Conv2D(128, (3,3), activation='relu', padding='same'),
    layer.GlobalAveragePooling2D(),      
    layer.Dropout(0.5),
    layer.Dense(128, activation='relu'),
    layer.Dense(3, activation='softmax')

])


# In[6]:


get_ipython().run_cell_magic('time', '', 'model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])\nhistory=model.fit(train_gen, epochs=100, callbacks=[early_stop], validation_data=val_gen)\n')


# In[7]:


test_gen = ImageDataGenerator(rescale = 1./255).flow_from_directory(
    "/home/ocelot/Desktop/CNN_RPS/Archive/test",
     target_size=(100,150),
     batch_size=32,
     class_mode="categorical",
     color_mode="grayscale",
     shuffle=False,
     seed=42)


# In[8]:


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


# In[9]:


testLoss, testAccuracy = model.evaluate(test_gen)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.layers as layer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
        class_mode="sparse",
        color_mode="grayscale",
        seed=42

)

val_gen = ImageDataGenerator(rescale = 1./255).flow_from_directory(
    "/home/ocelot/Desktop/CNN_RPS/Archive/validation",
     target_size=(100,150),
     batch_size=32,
     class_mode="sparse",
     color_mode="grayscale",
     seed=42)




# In[3]:


train_gen.batch_size = 3
fig, ax =  plt.subplots(1,3)
immagini = next(train_gen)
print(immagini[1][1])
for n,i in enumerate(immagini[0]):
    ax[n].imshow(i, cmap='gray')
train_gen.batch_size=32


# In[4]:


print(np.array([0.0, 0.0, 1.0]))


# In[5]:


early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
def crea_modello(drop_param = 0.5, learning_rate=0.001, num_filter_1=32, num_filter_2=64):
    model = tensorflow.keras.models.Sequential([
        layer.InputLayer(shape=(100,150,1)),
        layer.Conv2D(filters=(num_filter_1), kernel_size=(3,3), activation="relu"),
        layer.MaxPooling2D(pool_size=(2,2)),
        layer.Conv2D(filters=(num_filter_2), kernel_size=(3,3), activation="relu"),
        layer.MaxPooling2D(pool_size=(2,2)),
        layer.Flatten(),
        layer.Dense(128, activation="relu"),
        layer.Dropout(drop_param),
        layer.Dense(3, activation="softmax")
    ])
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model



# In[6]:


from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
model = KerasClassifier(model = crea_modello, epochs=15, batch_size=32,verbose=0)
griglia = {
    "model__drop_param": [0.25, 0.5],
    "model__learning_rate":[1e-3, 5e-4, 1e-4, 1e-2],
    "model__num_filter_1":[32,64],
    "model__num_filter_2":[32,64]
}

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=griglia,
    cv=kfold,               
    scoring='accuracy',         
    verbose=0,
    n_jobs=1 )


# In[7]:


import os
import cv2
Im = []
target = []

cat = ["paper", "scissors", "rock"]
vettori = {"paper":0, "scissors":2, "rock":1}
for i in cat:
    for j in os.listdir(os.path.join("/home/ocelot/Desktop/CNN_RPS/Archive/train", i)):
        Im.append(cv2.imread(os.path.join("/home/ocelot/Desktop/CNN_RPS/Archive/train", i,j), 0))
        target.append(vettori[i])
X_train = np.array(Im).astype('float32') / 255.0
X_train = np.expand_dims(X_train, axis=-1)
Y_train = np.array(target)


# In[8]:


get_ipython().run_cell_magic('time', '', 'risultati = grid_search.fit(X_train, Y_train )\n')


# In[9]:


risultati.best_params_


# In[10]:


model_def = crea_modello(drop_param=0.25, learning_rate=0.001, num_filter_1=32, num_filter_2=32)
history= model_def.fit(train_gen, epochs=100, callbacks=[early_stop], validation_data=val_gen)


# In[11]:


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


# In[12]:


test_gen = ImageDataGenerator(rescale = 1./255).flow_from_directory(
    "/home/ocelot/Desktop/CNN_RPS/Archive/test",
     target_size=(100,150),
     batch_size=32,
     class_mode="sparse",
     color_mode="grayscale",
     shuffle=False,
     seed=42)


# In[13]:


testLoss, testAccuracy = model_def.evaluate(test_gen)


# In[14]:


import numpy as np
from sklearn.metrics import classification_report
pred = model_def.predict(test_gen)
pred = np.argmax(pred, axis=1)
real_label = test_gen.classes
keys = test_gen.class_indices.keys()
keys = list(keys)
report = classification_report(real_label, pred, target_names=keys, output_dict=True)


# In[15]:


report["scissors"]


# In[16]:


report["rock"]


# In[17]:


report["paper"]


# In[18]:


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
    print("Label predetta: {} Label vera: {}".format(keys[pred[i]], keys[real_label[i]]))


# In[20]:


model_def.save("secondo.keras")


# In[ ]:





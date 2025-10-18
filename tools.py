


import numpy as np
from sklearn.metrics import classification_report
import cv2
import matplotlib.pyplot as plt



def create_split_gen(generator, path="", batch = 32, label_type="categorical", mode = "grayscale", seed = 42, shuffle = True):
    gen = generator.flow_from_directory(
        path,
        target_size=(100,150),
        batch_size=batch,
        class_mode=label_type,
        color_mode=mode,
        shuffle = shuffle,
        seed=seed)
    return gen





def plot_fitting_history(history):
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






def class_report(pred, generator):
    pred = np.argmax(pred, axis=1)
    real_label = generator.classes
    keys = generator.class_indices.keys()
    keys = list(keys)
    report = classification_report(real_label, pred, target_names=keys, output_dict=True)
    return report






def find_errors(pred, real, generator):
    pred = np.argmax(pred, axis=1)
    errori_indice=[]
    keys = list(generator.class_indices.keys())
    for i in range(len(real)):
        if real[i] != pred[i]:
            errori_indice.append(i)
    file = generator.filepaths
    errori = {"immagini":[], "LabelGiusta":[], "LabelPred":[]}
    for i in errori_indice:
        plt.figure(figsize=(4, 4))
        plt.imshow(cv2.cvtColor(cv2.imread(file[i]), cv2.COLOR_BGR2RGB))
        print("Label predetta: {} Label vera: {}".format(keys[pred[i]], keys[real[i]]))








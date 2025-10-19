# Repository for the project CNN_RPS for the course Statistical Methods for Machine Learning:
This repository contain the following files:
* image_pre_processing.ipynb: this notebook contain the code that was used for analyzing and processing the dataset
* primo_modello.ipynb, secondo_modello.ipynb, terzo_modello.ipynb: this notebooks contain the code for the definition, training and evaluation of the models used in the project
* tools.py: this file contains some function that were used by all the model files for creting the generators, visualizing the training history and evaluate the models.
* report.pdf: this file contain the report for the project
* For the image_pre_processing file, the path variable should be set to the directory that contains the scissors, rock, and paper folders from the original Kaggle dataset.
* For the model files, wherever /home/ocelot/Desktop/CNN_RPS/Archive/x appears, replace it with the path to the x_split folder of your processed dataset.
* After executing the image_pre_processing script,  in the directory specified by the path variable you will find three dataset splits: train, test, and val. Inside each split, there will be three folders named paper_clahe, scissors_clahe, and rock_clahe. To ensure compatibility with the model scripts, these folders must be renamed to paper, scissors, and rock respectively.

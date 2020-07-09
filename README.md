# K-Nearest-Neighbors-Anonymized-Data

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Libraries used](#libraries-used)
* [Dataset used](#dataset-used)
* [Built on](#built-on)
* [Model Training and Testing Steps](#model-training-and-testing-steps)
* [Ackowledgements](#ackowledgements)
* [Author](#author)


## About the Project 
The notebook __"02-K Nearest Neighbors Project"__ explores the basic use of __Pandas__ and will cover the basic commands of __Exploratory Data Analysis(EDA)__ to dive into the anonymized dataset. This dataset will be briefly analysed using some basic statistical tools and charts. The primary objective is to use the K Nearest Neighbors algorithm to correctly classify our data.</br></br>

It was required to scale our features before creating and training our model due to the fact that KNN works on euclidean distance measurement and scaling the variables makes sure they all contribute equally.

We first train our model on an arbitrary value of k for our KNN model. We then use the elbow method to be able to better select an appropriate k value for our algorithm. Finally, we train our model with our newly selected k value and evaluate it's performance using various metrics.

## Libraries used 
* Numpy
* Pandas
* Matplotlib
* Seaborn
* sklearn

```bash
import numpy as np                                                 # To implemennt milti-dimensional array and matrices
import pandas as pd                                                # For data manipulation and analysis
#import pandas_profiling
import matplotlib.pyplot as plt                                    # Plotting library for Python programming language and it's numerical mathematics extension NumPy
import seaborn as sns                                              # Provides high level informative statistical graphics
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
```

## Dataset used 
* __Perian Data__ - Anonymized Dataset

## Built with
* Jupyter Notebook

## Model Training and Testing Steps
1. Training the Model
2. Selecting arbitrary k value.
3. Predicting the Test Data
4. Evaluating the Model
5. Using elbow method to select better k value.
6. Examinnig and evaluating the new model.

## Ackowledgements
* <a href='http://www.pieriandata.com'>Perian Data</a> - Dataset

## Author - Sharan Sukesh





#TODO --cmap(y)-- doesnt work well with IRIS data 

## SOURCE -- https://github.com/rasbt/machine-learning-book/blob/main/ch03/ch03.py
# conda activate env2_det2
"""
- [important_notes_StandardScaler](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#plot-all-scaling-standard-scaler-section)
- more importantly, they can degrade the predictive performance of many machine learning algorithms. Unscaled data can also slow down or even prevent the convergence of many gradient-based estimators.
- more importantly that all features vary on comparable scales. 
- 
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#plot-all-scaling-standard-scaler-section

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler.transform
transform(X[, copy])
Perform standardization by centering and scaling.
"""
#
"""
-[sklearn.utils._bunch.Bunch](https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html#sklearn.utils.Bunch.keys)
# Load Iris from - sklearn -- as a <class 'sklearn.utils._bunch.Bunch'>
# https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html#sklearn.utils.Bunch.keys
"""

from utils_1 import *
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap


import matplotlib as mpl
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    minmax_scale,
)


def load_n_scale_data(data_name_str):
    """
    """
    if data_name_str == "iris":
        iris = datasets.load_iris()

        X_full = iris.data[:, [2, 3]]
        y_full = iris.target
        print("--[INFO]--type(iris)--",type(iris)) #--type(iris)-- <class 'sklearn.utils._bunch.Bunch'>
        print("--[INFO]--iris.keys()--",iris.keys()) # - dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
        #print("--[INFO]--type(iris.data)--",type(iris.data)) #-- <class 'numpy.ndarray'>
        #print("--[INFO]--type(iris.target)--",type(iris.target)) #-- <class 'numpy.ndarray'>
        #print("--[INFO]--type(iris.target_names)--",type(iris.target_names)) #
        print("--[INFO]--iris.target_names--",iris.target_names) # - ['setosa' 'versicolor' 'virginica']
        #print("--[INFO]--iris.frame--",iris.frame) # None
        #print("--[INFO]--iris.data_module--",iris.data_module) # sklearn.datasets.data
    
    elif data_name_str == "calif_housing":
        calif_housing = datasets.fetch_california_housing()
        X_full, y_full = calif_housing.data, calif_housing.target
        feature_names = calif_housing.feature_names
        print("[INFO]--calif_housing--feature_names--",feature_names)

    
    print('Class labels:', np.unique(y_full))
    #
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.3, random_state=1, stratify=y_full)
    #
    print('Labels counts in y:', np.bincount(y_full)) ##weights=w
    print('Labels counts in y_train:', np.bincount(y_train))
    print('Labels counts in y_test:', np.bincount(y_test))
    #
    _ = plt.hist(y_train, bins='auto')
    #plt.show()
    #
    # Standardizing the features:
    #
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return X_train_std , X_test_std , X_train , X_test , y_train, y_test

def train_n_pred_perceptron():
    """
    """
    # Training a perceptron via scikit-learn
    ppn = Perceptron(eta0=0.1, random_state=1)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    print('Misclassified examples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))


if __name__ == "__main__":
    data_name_str = "calif_housing"
    X_train_std , X_test_std , X_train , X_test , y_train, y_test = load_n_scale_data(data_name_str)
    # plots -- start - a_utils_plots_stdScalar
    plot_std_scalar()
    train_n_pred_perceptron()



# PDF == Probability density Function --->>  https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/random-variables-continuous/v/probability-density-functions
# PDF == Probability density Function --->> https://en.wikipedia.org/wiki/Probability_density_function
# that is, it is given by the area under the density function but above the horizontal axis and between the lowest and greatest values of the range.



"""
--features_idx---X.shape- (150, 2)
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/sklearn/preprocessing/_data.py:2583: UserWarning: n_quantiles (1000) is greater than the total number of samples (150). n_quantiles is set to n_samples.
  warnings.warn(
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/sklearn/preprocessing/_data.py:2583: UserWarning: n_quantiles (1000) is greater than the total number of samples (150). n_quantiles is set to n_samples.
  warnings.warn(

def generate_text(data_point)
"""

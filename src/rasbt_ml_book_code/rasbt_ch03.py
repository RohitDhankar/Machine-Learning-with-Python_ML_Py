## SOURCE -- https://github.com/rasbt/machine-learning-book/blob/main/ch03/ch03.py

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

# Load Iris from - sklearn -- as a <class 'sklearn.utils._bunch.Bunch'>
# https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html#sklearn.utils.Bunch.keys

iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target
print("--[INFO]--type(iris)--",type(iris)) #--type(iris)-- <class 'sklearn.utils._bunch.Bunch'>
print("--[INFO]--iris.keys()--",iris.keys()) # - dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
 
print("--[INFO]--type(iris.data)--",type(iris.data)) #-- <class 'numpy.ndarray'>
print("--[INFO]--type(iris.target)--",type(iris.target)) #-- <class 'numpy.ndarray'>
print("--[INFO]--type(iris.target_names)--",type(iris.target_names)) #
print("--[INFO]--iris.target_names--",iris.target_names) # - ['setosa' 'versicolor' 'virginica']
print("--[INFO]--iris.frame--",iris.frame) # None
print("--[INFO]--iris.data_module--",iris.data_module) # sklearn.datasets.data
print('Class labels:', np.unique(y))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


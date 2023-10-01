## SOURCE -- https://github.com/rasbt/machine-learning-book/blob/main/ch03/ch03.py
# conda activate env2_det2
"""
- [important_notes_StandardScaler](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#plot-all-scaling-standard-scaler-section)
- more importantly, they can degrade the predictive performance of many machine learning algorithms. Unscaled data can also slow down or even prevent the convergence of many gradient-based estimators.
- more importantly that all features vary on comparable scales. 
- 
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#plot-all-scaling-standard-scaler-section

- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler.transform
```
transform(X[, copy])
Perform standardization by centering and scaling.
```
"""
#
"""
-[sklearn.utils._bunch.Bunch](https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html#sklearn.utils.Bunch.keys)
# Load Iris from - sklearn -- as a <class 'sklearn.utils._bunch.Bunch'>
# https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html#sklearn.utils.Bunch.keys
"""


from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



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
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
#
print('Labels counts in y:', np.bincount(y)) ##weights=w
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))
#
_ = plt.hist(y_train, bins='auto')
plt.show()
#
# Standardizing the features:
#
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# PDF == Probability density Function --->>  https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/random-variables-continuous/v/probability-density-functions
# PDF == Probability density Function --->> https://en.wikipedia.org/wiki/Probability_density_function
# that is, it is given by the area under the density function but above the horizontal axis and between the lowest and greatest values of the range.





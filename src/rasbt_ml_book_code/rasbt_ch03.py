
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
#plt.show()
#
# Standardizing the features:
#
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#


## plots -- start 

feature_mapping = {
    "sepal_length": "sepal length (cm)",
    "sepal_width": "sepal width (cm)",
    "petal_length": 'petal length (cm)', 
    "petal_width": 'petal width (cm)'
}

# plasma does not exist in matplotlib < 1.5
cmap = getattr(cm, "plasma_r", cm.hot_r)

def create_axes(title, figsize=(16, 6)):
    """
    """

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # define the axis for the first plot
    left, width = 0.1, 0.22
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)

    # define the axis for the zoomed-in plot
    left = width + left + 0.2
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter_zoom = plt.axes(rect_scatter)
    ax_histx_zoom = plt.axes(rect_histx)
    ax_histy_zoom = plt.axes(rect_histy)

    # define the axis for the colorbar
    left, width = width + left + 0.13, 0.01

    rect_colorbar = [left, bottom, width, height]
    ax_colorbar = plt.axes(rect_colorbar)
    #plt.show() # OK Legends Only 

    return (
        (ax_scatter, ax_histy, ax_histx),
        (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
        ax_colorbar,
    )

def plot_distribution(axes, X, y, hist_nbins=50, title="", x0_label="", x1_label=""):

    """
    """
    print("[INFO]---plot_distribution-axes---",axes)
    print("[INFO]---plot_distribution-y---",y)

    ax, hist_X1, hist_X0 = axes
    ax.set_title(title)
    print("[INFO]---plot_distribution----title--",title)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)
    # The scatter plot
    colors = cmap(y) #TODO --cmap(y)-- doesnt work well with IRIS data 
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker="o", s=5, lw=0)#, c=colors)
    # Removing the top and the right spine for aesthetics
    # make nice axis layout
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    # Histogram for axis X1 (feature 5)
    hist_X1.set_ylim(ax.get_ylim())
    hist_X1.hist(
        X[:, 1], bins=hist_nbins, orientation="horizontal", color="grey", ec="grey"
    )
    hist_X1.axis("off")

    print("---plot_distribution---type(hist_X1---",type(hist_X1))

    # Histogram for axis X0 (feature 0)
    hist_X0.set_xlim(ax.get_xlim())
    hist_X0.hist(
        X[:, 0], bins=hist_nbins, orientation="vertical", color="grey", ec="grey"
    )
    hist_X0.axis("off")
    print("---plot_distribution---type(hist_X0---",type(hist_X0))
    ## <class 'matplotlib.axes._axes.Axes'>
    #plt.show() # OK_1


def make_plot(item_idx,X_full, y_full,distributions,features):
    """
    """
    print("--make_plot-making---")
    title, X = distributions[item_idx]
    print("--make_plot-----title-",title)

    ax_zoom_out, ax_zoom_in, ax_colorbar = create_axes(title)
    axarr = (ax_zoom_out, ax_zoom_in)
    plot_distribution(
        axarr[0],
        X,
        y,
        hist_nbins=200,
        x0_label=feature_mapping[features[0]],
        x1_label=feature_mapping[features[1]],
        title="Full data",
    )

    # zoom-in
    zoom_in_percentile_range = (0, 99)
    cutoffs_X0 = np.percentile(X[:, 0], zoom_in_percentile_range)
    cutoffs_X1 = np.percentile(X[:, 1], zoom_in_percentile_range)

    non_outliers_mask = np.all(X > [cutoffs_X0[0], cutoffs_X1[0]], axis=1) & np.all(
        X < [cutoffs_X0[1], cutoffs_X1[1]], axis=1
    )
    print("--make_plot-----non_outliers_mask-",type(non_outliers_mask))
    print("--make_plot-----non_outliers_mask.shape---",non_outliers_mask.shape)

    plot_distribution(
        axarr[1],
        X[non_outliers_mask],
        y[non_outliers_mask],
        hist_nbins=50,
        x0_label=feature_mapping[features[0]],
        x1_label=feature_mapping[features[1]],
        title="Zoom-in",
    )

    norm = mpl.colors.Normalize(y_full.min(), y_full.max())
    mpl.colorbar.ColorbarBase(
        ax_colorbar,
        cmap=cmap,
        norm=norm,
        orientation="vertical",
        label="Color mapping for values of y",
    )


def plot_std_scalar():
    """
    """
    print("[INFO]--type(iris)--",type(iris.target)) #<class 'sklearn.utils._bunch.Bunch'>
    print("[INFO]--iris.keys()--",iris.keys()) # -iris.keys()-- dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
    #print("[INFO]--iris.keys()--",iris.frame) # always None ? 
    X_full, y_full = iris.data, iris.target

    feature_names = iris.feature_names
    print("[INFO]--iris---feature_names--",feature_names) 
    #[INFO]--iris---feature_names-- ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']



    features = ["sepal_length", "petal_width"]
    features_idx = [feature_names.index(feature_mapping[feature]) for feature in features]
    print("--features_idx--",features_idx) #[0, 3]
    print("--features_idx----X_full.shape-",X_full.shape) #---X_full.shape- (150, 4)
    X = X_full[:, features_idx]

    print("--features_idx---type(X)-",type(X))
    print("--features_idx---X.shape-",X.shape) #--features_idx----X.shape- (150, 2)


    distributions = [
        ("Unscaled data", X),
        ("Data after standard scaling", StandardScaler().fit_transform(X)),
        ("Data after min-max scaling", MinMaxScaler().fit_transform(X)),
        ("Data after max-abs scaling", MaxAbsScaler().fit_transform(X)),
        (
            "Data after robust scaling",
            RobustScaler(quantile_range=(25, 75)).fit_transform(X),
        ),
        (
            "Data after power transformation (Yeo-Johnson)",
            PowerTransformer(method="yeo-johnson").fit_transform(X),
        ),
        (
            "Data after power transformation (Box-Cox)",
            PowerTransformer(method="box-cox").fit_transform(X),
        ),
        # (
        #     "Data after quantile transformation (uniform pdf)",
        #     QuantileTransformer(
        #         output_distribution="uniform", random_state=42
        #     ).fit_transform(X),
        # ),
        # (
        #     "Data after quantile transformation (gaussian pdf)",
        #     QuantileTransformer(
        #         output_distribution="normal", random_state=42
        #     ).fit_transform(X),
        # ),
        ("Data after sample-wise L2 normalizing", Normalizer().fit_transform(X)),
    ]

    # scale the output between 0 and 1 for the colorbar
    y = minmax_scale(y_full)
    ls_item_idx = [0,1,2,3,4]
    for iter_idx in range(len(ls_item_idx)):
        make_plot(iter_idx,X_full, y_full,distributions,features)
        plt.show()


plot_std_scalar()

# ## Training a perceptron via scikit-learn
# ppn = Perceptron(eta0=0.1, random_state=1)
# ppn.fit(X_train_std, y_train)
# y_pred = ppn.predict(X_test_std)
# print('Misclassified examples: %d' % (y_test != y_pred).sum())
# print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
# print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))












# PDF == Probability density Function --->>  https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/random-variables-continuous/v/probability-density-functions
# PDF == Probability density Function --->> https://en.wikipedia.org/wiki/Probability_density_function
# that is, it is given by the area under the density function but above the horizontal axis and between the lowest and greatest values of the range.




"""
--features_idx---X.shape- (150, 2)
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/sklearn/preprocessing/_data.py:2583: UserWarning: n_quantiles (1000) is greater than the total number of samples (150). n_quantiles is set to n_samples.
  warnings.warn(
/home/dhankar/anaconda3/envs/env2_det2/lib/python3.9/site-packages/sklearn/preprocessing/_data.py:2583: UserWarning: n_quantiles (1000) is greater than the total number of samples (150). n_quantiles is set to n_samples.
  warnings.warn(


"""
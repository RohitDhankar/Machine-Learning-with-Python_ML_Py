


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


def make_plot(item_idx,X_full, y_full,y_min_max_scaled,distributions,features):
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
        y_min_max_scaled,
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
        y_min_max_scaled[non_outliers_mask],
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
    y_min_max_scaled = minmax_scale(y_full) ## Original code calls this VAR -- y_min_max_scaled -- as SIMPLY - small y 
    ls_item_idx = [0,1,2,3,4]
    for iter_idx in range(len(ls_item_idx)):
        make_plot(iter_idx,X_full,y_full,y_min_max_scaled,distributions,features)
        plt.show()

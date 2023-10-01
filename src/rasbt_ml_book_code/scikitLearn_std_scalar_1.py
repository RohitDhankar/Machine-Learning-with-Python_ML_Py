
#ORIGINAL SOURCE - https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#plot-all-scaling-standard-scaler-section

# Author:  Raghav RV <rvraghav93@gmail.com>
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Thomas Unterthiner
# License: BSD 3 clause

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

df_calif_h = fetch_california_housing()

print("[INFO]--type(df_calif_h)--",type(df_calif_h)) #<class 'sklearn.utils._bunch.Bunch'>
print("[INFO]--df_calif_h.keys()--",df_calif_h.keys()) #dict_keys(['data', 'target', 'frame', 'target_names', 'feature_names', 'DESCR'])
print("[INFO]--df_calif_h.keys()--",df_calif_h.frame) # None

X_full, y_full = df_calif_h.data, df_calif_h.target
feature_names = df_calif_h.feature_names
print("[INFO]--df_calif---feature_names--",feature_names) # - ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

# feature_mapping = {
#     "MedInc": "Median income in block",
#     "HouseAge": "Median house age in block",
#     "AveRooms": "Average number of rooms",
#     "AveBedrms": "Average number of bedrooms",
#     "Population": "Block population",
#     "AveOccup": "Average house occupancy",
#     "Latitude": "House block latitude",
#     "Longitude": "House block longitude",
# }

# # Take only 2 features to make visualization easier
# # Feature MedInc has a long tail distribution.
# # Feature AveOccup has a few but very large outliers.

# features = ["MedInc", "AveOccup"]
# features_idx = [feature_names.index(feature) for feature in features]
# X = X_full[:, features_idx]
# distributions = [
#     ("Unscaled data", X),
#     ("Data after standard scaling", StandardScaler().fit_transform(X)),
#     ("Data after min-max scaling", MinMaxScaler().fit_transform(X)),
#     ("Data after max-abs scaling", MaxAbsScaler().fit_transform(X)),
#     (
#         "Data after robust scaling",
#         RobustScaler(quantile_range=(25, 75)).fit_transform(X),
#     ),
#     (
#         "Data after power transformation (Yeo-Johnson)",
#         PowerTransformer(method="yeo-johnson").fit_transform(X),
#     ),
#     (
#         "Data after power transformation (Box-Cox)",
#         PowerTransformer(method="box-cox").fit_transform(X),
#     ),
#     (
#         "Data after quantile transformation (uniform pdf)",
#         QuantileTransformer(
#             output_distribution="uniform", random_state=42
#         ).fit_transform(X),
#     ),
#     (
#         "Data after quantile transformation (gaussian pdf)",
#         QuantileTransformer(
#             output_distribution="normal", random_state=42
#         ).fit_transform(X),
#     ),
#     ("Data after sample-wise L2 normalizing", Normalizer().fit_transform(X)),
# ]

# # scale the output between 0 and 1 for the colorbar
# y = minmax_scale(y_full)

# # plasma does not exist in matplotlib < 1.5
# cmap = getattr(cm, "plasma_r", cm.hot_r)


# def create_axes(title, figsize=(16, 6)):
#     fig = plt.figure(figsize=figsize)
#     fig.suptitle(title)

#     # define the axis for the first plot
#     left, width = 0.1, 0.22
#     bottom, height = 0.1, 0.7
#     bottom_h = height + 0.15
#     left_h = left + width + 0.02

#     rect_scatter = [left, bottom, width, height]
#     rect_histx = [left, bottom_h, width, 0.1]
#     rect_histy = [left_h, bottom, 0.05, height]

#     ax_scatter = plt.axes(rect_scatter)
#     ax_histx = plt.axes(rect_histx)
#     ax_histy = plt.axes(rect_histy)

#     # define the axis for the zoomed-in plot
#     left = width + left + 0.2
#     left_h = left + width + 0.02

#     rect_scatter = [left, bottom, width, height]
#     rect_histx = [left, bottom_h, width, 0.1]
#     rect_histy = [left_h, bottom, 0.05, height]

#     ax_scatter_zoom = plt.axes(rect_scatter)
#     ax_histx_zoom = plt.axes(rect_histx)
#     ax_histy_zoom = plt.axes(rect_histy)

#     # define the axis for the colorbar
#     left, width = width + left + 0.13, 0.01

#     rect_colorbar = [left, bottom, width, height]
#     ax_colorbar = plt.axes(rect_colorbar)

#     return (
#         (ax_scatter, ax_histy, ax_histx),
#         (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
#         ax_colorbar,
#     )


# def plot_distribution(axes, X, y, hist_nbins=50, title="", x0_label="", x1_label=""):
#     ax, hist_X1, hist_X0 = axes

#     ax.set_title(title)
#     ax.set_xlabel(x0_label)
#     ax.set_ylabel(x1_label)

#     # The scatter plot
#     colors = cmap(y)
#     ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker="o", s=5, lw=0, c=colors)

#     # Removing the top and the right spine for aesthetics
#     # make nice axis layout
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.get_xaxis().tick_bottom()
#     ax.get_yaxis().tick_left()
#     ax.spines["left"].set_position(("outward", 10))
#     ax.spines["bottom"].set_position(("outward", 10))

#     # Histogram for axis X1 (feature 5)
#     hist_X1.set_ylim(ax.get_ylim())
#     hist_X1.hist(
#         X[:, 1], bins=hist_nbins, orientation="horizontal", color="grey", ec="grey"
#     )
#     hist_X1.axis("off")

#     # Histogram for axis X0 (feature 0)
#     hist_X0.set_xlim(ax.get_xlim())
#     hist_X0.hist(
#         X[:, 0], bins=hist_nbins, orientation="vertical", color="grey", ec="grey"
#     )
#     hist_X0.axis("off")
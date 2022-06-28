import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, make_union
from packages.sklearn_xarray import Stacker, Select

#!pip install sklearn_xarray
from sklearn_xarray.datasets import load_dummy_dataarray
from sklearn_xarray import wrap
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.datasets import make_gaussian_quantiles

# import plotly



# # construct dataset
# X1, y1 = make_gaussian_quantiles(cov=1., n_samples=1000, n_features=3, n_classes=2, random_state=1)
# X1 = pd.DataFrame(X1, columns=['x', 'y', 'z'])
# y1 = pd.Series(y1)
# ax = plt.axes(projection='3d')
# # Data for a three-dimensional line
# # zline = np.linspace(0, 15, 1000)
# # xline = np.sin(zline)
# # yline = np.cos(zline)
# # ax.plot3D(xline, yline, zline, 'gray')
#
# # Data for three-dimensional scattered points
# zdata = X1.z  # 15 * np.random.random(100)
# xdata = X1.x  # np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = X1.y  # np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

#==================#

# X, y = make_classification(n_samples=1000, n_features=3, n_informative=3,
#                           n_redundant=0, n_repeated=0, n_classes=3,
#                           n_clusters_per_class=2, class_sep=1.5,
#                           flip_y=0, weights=[0.5, 0.5, 0.5])
#
# X = pd.DataFrame(X)
# y = pd.Series(y)

#====================#
# X = load_dummy_dataarray()
# Xt = wrap(StandardScaler()).fit_transform(X)

#====================#

# from sklearn-xarray.readthedocs.io

# Make synthetic data
lat, lon = np.ogrid[-45:45:50j, 0:360:100j]
noise = np.random.randn(lat.shape[0], lon.shape[1])

data_vars = {
    'a': (['lat', 'lon'], np.sin(lat/90 + lon/100)),
    'b': (['lat', 'lon'], np.cos(lat/90 + lon/100)),
    'noise': (['lat', 'lon'], noise)
}

coords = {'lat': lat.ravel(), 'lon': lon.ravel()}
dataset = xr.Dataset(data_vars, coords)

# make a simple linear model for the output
# y = a + 0.5b + 1

x = dataset[['a', 'b']]
y = dataset.a + dataset.b * 0.5 + 0.3 * dataset.noise + 1
y.plot()
plt.show()

# now we want to fit a linear regression model using these data

mod = make_pipeline(
    make_union(
        make_pipeline(Select('a'), Stacker()),
        make_pipeline(Select('b'), Stacker()),
    ),
    LinearRegression()
)

# for now we have to use Stacker manually to transform the output data into a 2d array

y_np = Stacker().fit_transform(y)
print(y_np)

# fit the model

mod.fit(x, y_np)
# print the coefficients
lm = mod.named_steps['linearregression']
coefs = tuple(lm.coef_.flat)
print("The exact regression model is y=1+a_0.5b + noise")
print("The estimated coefficients are a: {}, b: {}".format(*coefs))
print(f"The estimated intercept is {lm.intercept_[0]}")
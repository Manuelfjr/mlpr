# MLPR Library
[![TestPyPI version](https://img.shields.io/badge/dynamic/json?url=https://test.pypi.org/pypi/mlpr/json&label=TestPyPI%20version&query=$.info.version&color=blue)](https://test.pypi.org/project/mlpr/)

<!-- <img src="assets/regression_plots.png" alt="Regression Plots" href="assets/regression_plots.png"> -->
<!-- <a href="https://ibb.co/dp1TxTq"><img src="https://i.ibb.co/s1kfzfG/regression-plots.png" alt="regression-plots" border="0"></a> -->
<a href="https://ibb.co/T2tydfX"><img src="https://i.ibb.co/6wFQLMh/regression-plots.png" alt="regression-plots" border="0"></a>

This repository is a developing library named `MLPR` (Machine Learning Pipeline Report). It aims to facilitate the creation of machine learning models in various areas such as regression, forecasting, classification, and clustering. The library allows the user to perform tuning of these models, generate various types of plots for post-modeling analysis, and calculate various metrics.

In addition, `MLPR` allows the creation of metric reports using [`Jinja2`](https://pypi.org/project/Jinja2/), including the obtained graphs. The user can customize the report template according to their needs.

# Using the MLPR Library
The MLPR library is a powerful tool for machine learning and data analysis. Here's a brief guide on how to use it.

## Installation
Before you start, make sure you have installed the MLPR library. You can do this by running pip install mlpr.


```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mlpr
```

<!-- # Regression

The example of how to use the module `mlpr.ml.regression` can be view in [`notebooks/regression/`](/notebooks/regression/) -->

# Regression

How to use the module for regression problems.

## Importing the Library
First, import the necessary modules from the library:

```python
from mlpr.ml.regression import metrics, plots
from mlpr.ml.tunning.grid_search import GridSearch
from mlpr.reports.reports import ReportGenerator
```

## Loading the Data
Load your dataset. In this example, we're generating a dataset for regression using sklearn:

```python
n_feats = 11
n_instances = 1000
n_invert = 50
n_noise = 20
```

```python
X, y = make_regression(n_samples=n_instances, n_features=n_feats, noise=n_noise)

# introduce of noises
indices = np.random.choice(y.shape[0], size=n_invert, replace=False)
y[indices] = np.max(y) - y[indices]

data = pd.DataFrame(data=X, columns=[f'feature_{i}' for i in range(1, n_feats + 1)])
data['target'] = y
```

## Set the seed
Set the random seed for reproducibility

```python
n_seed = 42
np.random.seed(n_seed)
```

## Preparing the Data
Split your data into features ($X$) and target ($y$):

```python
X = data.drop("target", axis=1)
y = data["target"].values
```
## Model Training
Define the parameters for your models and use `GridSearch` to find the best model:

```python
models_params = {
    Ridge: {
        'alpha': [1.0, 10.0, 15, 20],
        'random_state': [n_seed]
    },
    Lasso: {
        'alpha': [0.1, 1.0, 10.0],
        'random_state': [n_seed]
    },
    SVR: {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf']
    },
    RandomForestRegressor: {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 5, 10],
        'random_state': [n_seed]
    },
    GradientBoostingRegressor: {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.05, 0.01],
        'random_state': [n_seed]
    },
    XGBRegressor: {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.05, 0.01],
        'random_state': [n_seed]
    }
}

params_split = {
    'test_size': 0.25,
    'random_state': n_seed
}
params_norm = {'with_mean': True, 'with_std': True}

grid_search = GridSearch(
    X,
    y,
    params_split=params_split,
    models_params=models_params,
    normalize=True,
    params_norm=params_norm
)
grid_search.search(cv=5, n_jobs=-1)

best_model, best_params = \
    grid_search \
    .get_best_model()

```


## Making Predictions
Use the best model to make predictions:

```python
data_train["y_pred"] = \
    grid_search \
        .best_model \
            .predict(grid_search.X_train)
```

## Evaluating the Model
Calculate various metrics to evaluate the performance of the model:

```python
rm = metrics.RegressionMetrics(
    data_train,
    *["y_true", "y_pred"]
)
results = rm.calculate_metrics(
    ["mape", "rmse", "kolmogorov_smirnov", "confusion_matrix", "calculate_kappa"],
    {
        "mape": {},
        "rmse": {},
        "kolmogorov_smirnov": {},
        "confusion_matrix": {"n_bins": k},
        "calculate_kappa": {"n_bins": k}
    }
)
```

## Results

The output it's a dictionary object with the calculated metrics, like this:

```
{'mape': 39.594540526956436,
 'rmse': 54.09419440169204,
 'kolmogorov_smirnov': (0.1510574018126888, 0.0010310446878578096),
 'confusion_matrix': (array([[54, 57,  2,  0],
         [16, 70, 21,  0],
         [ 0, 37, 37,  3],
         [ 0,  6, 25,  3]]),
  {'precision': array([0.77142857, 0.41176471, 0.43529412, 0.5       ]),
   'recall': array([0.47787611, 0.65420561, 0.48051948, 0.08823529]),
   'f1_score': array([0.59016393, 0.50541516, 0.45679012, 0.15      ]),
   'support': array([113, 107,  77,  34]),
   'accuracy': 0.4954682779456193}),
 'calculate_kappa': {0: {'confusion_matrix': array([[202,  16],
          [ 59,  54]]),
   'kappa_score': 0.4452885840055415,
   'metrics': {'precision': array([0.77394636, 0.77142857]),
    'recall': array([0.9266055 , 0.47787611]),
    'f1_score': array([0.8434238 , 0.59016393]),
    'support': array([218, 113]),
    'accuracy': 0.7734138972809668}},
  1: {'confusion_matrix': array([[124, 100],
          [ 37,  70]]),
   'kappa_score': 0.180085703437178,
   'metrics': {'precision': array([0.77018634, 0.41176471]),
    'recall': array([0.55357143, 0.65420561]),
    'f1_score': array([0.64415584, 0.50541516]),
    'support': array([224, 107]),
    'accuracy': 0.5861027190332326}},
  2: {'confusion_matrix': array([[206,  48],
          [ 40,  37]]),
   'kappa_score': 0.2813579394059016,
   'metrics': {'precision': array([0.83739837, 0.43529412]),
    'recall': array([0.81102362, 0.48051948]),
    'f1_score': array([0.824     , 0.45679012]),
    'support': array([254,  77]),
    'accuracy': 0.7341389728096677}},
  3: {'confusion_matrix': array([[294,   3],
          [ 31,   3]]),
   'kappa_score': 0.12297381546134645,
   'metrics': {'precision': array([0.90461538, 0.5       ]),
    'recall': array([0.98989899, 0.08823529]),
    'f1_score': array([0.94533762, 0.15      ]),
    'support': array([297,  34]),
    'accuracy': 0.8972809667673716}}}}
```



## Visualizing the Results
Plot the results using the `RegressionPlots` module:

```python
rp = \
    plots \
        .RegressionPlots(
            data_train,
            color_palette=["#FF4B3E", "#1C2127"]
        )
```

```py
fig, axs = rp.grid_plot(
    plot_functions=[
        ['graph11', 'graph12', 'graph13'],
        ['graph21', 'graph22'],
        ['graph23']
    ],
    plot_args={
        'graph11': {
            "plot": "scatter",
            "params": {
                'y_true_col': 'y_true',
                'y_pred_col': 'y_pred',
                'linecolor': '#1C2127',
                'worst_interval': True,
                'metrics': rm.metrics["calculate_kappa"],
                'class_interval': rm._class_intervals,
                'method': 'recall',
                'positive': True
            }
        },
        'graph12': {
            "plot": "plot_ecdf",
            "params": {
                'y_true_col': 'y_true',
                'y_pred_col': 'y_pred'
            }
        },
        'graph21': {
            "plot": "plot_kde",
            "params": {
                'columns': ['y_true', 'y_pred']
            }
        },
        'graph22': {
            "plot": "plot_error_hist",
            "params": {
                'y_true_col': 'y_true',
                'y_pred_col': 'y_pred',
                'linecolor': '#1C2127'
            }
        },
        'graph13': {
            "plot": "plot_fitted",
            "params": {
                'y_true_col': 'y_true',
                'y_pred_col': 'y_pred',
                'condition': (data_train["y_true"] >= 424.5132071505618),
                'sample_size': None
            }
        },
        'graph23': {
            "plot": "plot_fitted",
            "params": {
                'y_true_col': 'y_true',
                'y_pred_col': 'y_pred',
                'condition': None,
                'sample_size': None
            }
        },
    },
    show_inline=True
)
```

<a href="https://ibb.co/3pqSmnZ"><img src="https://i.ibb.co/Jv4rj9J/regression-plots.png" alt="regression-plots" border="0"></a>


## Reports

Here you can see the <a href="https://raw.githack.com/Manuelfjr/mlpr/develop/data/05_reports/report_model.html">report</a> output.

# Contact

Here you can find my contact information:

<div align="center">
    <a href="https://github.com/Manuelfjr/mlpr">
        <img src="https://avatars.githubusercontent.com/u/53409857?v=4" alt="Profile Image" width="200" height="200" style="border-radius: 50%;">
    </a>
</div>

<br>

<div align="center">
<a href="https://manuelfjr.github.io"><img src="https://img.icons8.com/ios/30/000000/domain.png"/></a>
<a href="https://github.com/manuelfjr"><img src="https://img.icons8.com/ios/30/000000/github--v1.png"/></a>
<a href="https://www.instagram.com/manuelferreirajr/"><img src="https://img.icons8.com/ios/30/000000/instagram-new--v1.png"/></a>
<a href="https://www.linkedin.com/in/manuefjr/"><img src="https://img.icons8.com/ios/30/000000/linkedin.png"/></a>
<a href="mailto:ferreira.jr.ufpb@gmail.com"><img src="https://img.icons8.com/ios/30/000000/email.png"/></a>
</div>

# License

This project is licensed under the terms of the MIT license. For more details, see the [LICENSE](/LICENSE) file in the project's root directory.
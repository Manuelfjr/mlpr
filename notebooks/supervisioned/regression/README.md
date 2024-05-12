# Regression

How to use the module for regression problems.

## Importing the Library
First, import the necessary modules from the library:

```python
from mlpr.ml.supervisioned.regression import metrics, plots
from mlpr.ml.supervisioned.tunning.grid_search import GridSearch
from mlpr.reports.create import ReportGenerator
```

## Loading the Data
Load your dataset. In this example, we're generating a dataset for regression using sklearn:

```python
n_feats = 11
n_instances = 1000
n_invert = 50
n_noise = 20
cv = 10
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
        'alpha': [1.0, 10.0, 15., 20.],
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

params_split: dict[str, float | int] = {
    'test_size': 0.25,
    'random_state': n_seed
}
params_norm: dict[str, bool] = {'with_mean': True, 'with_std': True}

grid_search = GridSearch(
    X,
    y,
    params_split=params_split,
    models_params=models_params,
    normalize=True,
    scoring='neg_mean_squared_error',
    metrics={'neg_mean_squared_error': rmse},
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
data_train = pd.DataFrame(
    grid_search.X_train,
    columns=X.columns
)
data_train["y_true"] = grid_search.y_train
data_train["y_pred"] = grid_search.best_model.predict(grid_search.X_train)
```

## Evaluating the Model
Calculate various metrics to evaluate the performance of the model:

```python
k = 3
```

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
                'condition': (
                    (
                        rm._worst_interval_kappa[0] <= data_train["y_true"]
                    ) & (
                        data_train["y_true"] <= rm._worst_interval_kappa[1]
                    )
                ),
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

[![fig](/assets/regression_plots.png)](/assets/regression_plots.png)


## Reports

Here you can see the <a href="https://raw.githack.com/Manuelfjr/mlpr/develop/data/05_reports/report_model.html">report</a> output.

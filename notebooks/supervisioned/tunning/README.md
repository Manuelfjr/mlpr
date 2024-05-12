# MLPR for model selection

MLPR used for model selection.


## Importing the Library
First, import the necessary modules from the library:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score


from mlpr.ml.supervisioned.tunning.grid_search import GridSearch
from utils.reader import read_file_yaml
```

## Methods

Here we have a custom method for accuracy to use in model selection. Thus:

```python
def custom_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred, normalize=False)
```

## Set parameters

```python
n_samples = 1000
centers = [(0, 0), (3, 4.5)]
n_features = 2
cluster_std = 1.3
random_state = 42
cv = 5
```

```python
np.random.seed(random_state)
```

```python
params_split: dict[str, float | int] = {
    'test_size': 0.25,
    'random_state': random_state
}
params_norm: dict[str, bool] = {'with_mean': True, 'with_std': True}
model_metrics: dict[str, any] = {
    'custom_accuracy': custom_accuracy_score,
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'kappa': cohen_kappa_score,
    'f1': f1_score,
}
```

## Loading the Data
Load your dataset. In this example, we're generating a dataset for classification using sklearn:


```python
X, y = make_blobs(
    n_samples=n_samples,
    centers=centers,
    n_features=n_features,
    cluster_std=cluster_std,
    random_state=random_state
)
```

## Plot the dataset

```python
fig, ax = plt.subplots(1, 1, figsize=(14, 6))

ax.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
ax.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
ax.set_title("Dataset")
ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
```

![fig0](/assets/tunning_scatter.png)

## Cross-validtion

```python
models: dict[BaseEstimator, dict] = {
    RandomForestClassifier: {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'random_state': [random_state]
    },
    GradientBoostingClassifier: {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.05, 0.01, 0.005],
        'subsample': [0.5, 0.8, 1.0],
        'random_state': [random_state]
    },
    LogisticRegression: {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'random_state': [random_state]
    },
    GaussianNB: {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    },
    KNeighborsClassifier: {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    SVC: {
        'C': [0.01, 0.1, 1.0, 10.0],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto'],
        'random_state': [random_state]
    },
    DecisionTreeClassifier: {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'random_state': [random_state]
    }
}
```

```python
grid_search = GridSearch(
    X,
    y,
    params_split=params_split,
    models_params=models,
    normalize=True,
    params_norm=params_norm,
    scoring='accuracy',
    metrics=model_metrics
)
grid_search.search(cv=cv, n_jobs=-1)

best_model, best_params = \
    grid_search \
    .get_best_model()
```

```python
results: pd.DataFrame = pd.DataFrame(grid_search._metrics).T
results
```

<table>
    <tr>
        <th></th>
        <th>custom_accuracy</th>
        <th>accuracy</th>
        <th>precision</th>
        <th>recall</th>
        <th>kappa</th>
        <th>f1</th>
    </tr>
    <tr>
        <td>RandomForestClassifier</td>
        <td>246.0</td>
        <td>0.984</td>
        <td>0.983607</td>
        <td>0.983607</td>
        <td>0.967982</td>
        <td>0.983607</td>
    </tr>
    <tr>
        <td>GradientBoostingClassifier</td>
        <td>244.0</td>
        <td>0.976</td>
        <td>0.975410</td>
        <td>0.975410</td>
        <td>0.951972</td>
        <td>0.975410</td>
    </tr>
    <tr>
        <td>LogisticRegression</td>
        <td>245.0</td>
        <td>0.980</td>
        <td>0.983471</td>
        <td>0.975410</td>
        <td>0.959969</td>
        <td>0.979424</td>
    </tr>
    <tr>
        <td>GaussianNB</td>
        <td>244.0</td>
        <td>0.976</td>
        <td>0.975410</td>
        <td>0.975410</td>
        <td>0.951972</td>
        <td>0.975410</td>
    </tr>
    <tr>
        <td>KNeighborsClassifier</td>
        <td>245.0</td>
        <td>0.980</td>
        <td>0.983471</td>
        <td>0.975410</td>
        <td>0.959969</td>
        <td>0.979424</td>
    </tr>
    <tr>
        <td>SVC</td>
        <td>245.0</td>
        <td>0.980</td>
        <td>0.983471</td>
        <td>0.975410</td>
        <td>0.959969</td>
        <td>0.979424</td>
    </tr>
    <tr>
        <td>DecisionTreeClassifier</td>
        <td>242.0</td>
        <td>0.968</td>
        <td>0.991379</td>
        <td>0.942623</td>
        <td>0.935889</td>
        <td>0.966387</td>
    </tr>
</table>

## Best model
Here we can see the distribution for the best classifier.

```python
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
ax[0].plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
ax[0].set_title("Dataset")
ax[0].set_frame_on(False)
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[1].plot(
    grid_search.X_test[:, 0][grid_search.y_test == 0],
    grid_search.X_test[:, 1][grid_search.y_test == 0],
    "bs"
)
ax[1].plot(
    grid_search.X_test[:, 0][grid_search.y_test == 1],
    grid_search.X_test[:, 1][grid_search.y_test == 1],
    "g^"
)
ax[1].set_title(grid_search.best_model.__class__.__name__)
ax[1].set_frame_on(False)
ax[1].set_xticks([])
ax[1].set_yticks([])
fig.tight_layout()
```


[![fig1](/assets/tunning_best_model.png)](/assets/tunning_best_model.png)



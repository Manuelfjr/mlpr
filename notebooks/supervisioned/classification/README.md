# Classification: uncertainty estimation

How to use the module for uncertainty estimation in classification tasks.

## Importing the Library
First, import the necessary modules from the library:

```python
from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.datasets import make_blobs
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             precision_score, recall_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from mlpr.ml.supervisioned.classification.uncertainty import UncertaintyPlots
from mlpr.ml.supervisioned.classification.utils import calculate_probas
from mlpr.ml.supervisioned.tunning.grid_search import GridSearch

import warnings
warnings.filterwarnings("ignore")
```

## Parameters
Setting parameters for the experiments.

```python
random_state: int = 42
n_feats: int = 2
n_size: int = 1000
centers: list[tuple] = [
    (0, 2),
    (2, 0),
    (5, 4.5)
]
n_class: int = len(centers)
cluster_std: list[float] = [1.4, 1.4, 0.8]
cv: int = 5
np.random.seed(random_state)
```

```python
params: dict[str, dict[str, Any]] = {
    "n_samples": n_size,
    "n_features": n_feats,
    "centers": centers,
    "cluster_std": cluster_std,
    "random_state": random_state
}
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

model_metrics: dict[str, Any] = {
    'custom_accuracy': partial(accuracy_score, normalize=False),
    'accuracy': accuracy_score,
    'precision': partial(precision_score, average='macro'),
    'recall': partial(recall_score, average='macro'),
    'kappa': cohen_kappa_score,
    'f1': partial(f1_score, average='macro'),
}
```

## Load the dataset

Here we are generating a dataset for experiments, using blobs from scikit-learn.

```python
X, y = make_blobs(
    **params
)
```

## Plot the dataset
Behavior of the dataset used in the experiment.

```python
markers = ['o', 'v', '^']
fig, ax = plt.subplots(1, 1, figsize=(14, 6))

colors = generate_colors("FF4B3E", "1C2127", len(np.unique(y)))

for i, k in enumerate(np.unique(y)):
  ax.scatter(X[:, 0][y == k], X[:, 1][y == k], marker=markers[i % len(markers)], color=colors[i], label=f"c{i}")

ax.set_title("Dataset")
ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])
for i, (center, color) in enumerate(zip(centers, colors)):
    ax.scatter(
        center[0],
        center[1],
        color="white",
        linewidths=3,
        marker="o",
        edgecolor="black",
        s=120,
        label="center" if i == 0 else None
    )
plt.legend()
fig.tight_layout()
```

[![fig1](/assets/classification_scatter.png)](/assets/classification_scatter.png)

## Cross-validation

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
        'random_state': [random_state],
        'max_iter': [10000]
    },
    GaussianNB: {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    },
    SVC: {
        'C': [0.01, 0.1, 1.0, 10.0],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto'],
        'probability': [True],
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
    <thead>
        <tr>
            <th></th>
            <th>custom_accuracy</th>
            <th>accuracy</th>
            <th>precision</th>
            <th>recall</th>
            <th>kappa</th>
            <th>f1</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>RandomForestClassifier</td>
            <td>222.0</td>
            <td>0.888</td>
            <td>0.883566</td>
            <td>0.885902</td>
            <td>0.831666</td>
            <td>0.882724</td>
        </tr>
        <tr>
            <td>GradientBoostingClassifier</td>
            <td>221.0</td>
            <td>0.884</td>
            <td>0.878830</td>
            <td>0.881207</td>
            <td>0.825583</td>
            <td>0.878421</td>
        </tr>
        <tr>
            <td>LogisticRegression</td>
            <td>230.0</td>
            <td>0.920</td>
            <td>0.915170</td>
            <td>0.916987</td>
            <td>0.879457</td>
            <td>0.915662</td>
        </tr>
        <tr>
            <td>GaussianNB</td>
            <td>231.0</td>
            <td>0.924</td>
            <td>0.919375</td>
            <td>0.921682</td>
            <td>0.885539</td>
            <td>0.920046</td>
        </tr>
        <tr>
            <td>SVC</td>
            <td>230.0</td>
            <td>0.920</td>
            <td>0.915170</td>
            <td>0.916987</td>
            <td>0.879457</td>
            <td>0.915662</td>
        </tr>
        <tr>
            <td>DecisionTreeClassifier</td>
            <td>214.0</td>
            <td>0.856</td>
            <td>0.848155</td>
            <td>0.845622</td>
            <td>0.782325</td>
            <td>0.846414</td>
        </tr>
    </tbody>
</table>

## Probabilities

Getting probabilities for uncertainty generation.

```python
probas = calculate_probas(grid_search.fitted, grid_search.X_train)
```

## Plot best model result

Plotting the result for best model.

```python
up = UncertaintyPlots()
```

```python
fig_un, ax_un = up.uncertainty(
    model_names=[[best_model.__class__.__name__]],
    probs={best_model.__class__.__name__: probas[best_model.__class__.__name__]},
    X=grid_search.X_train,
    figsize=(20, 6),
    cmap='RdYlGn',
    show_inline=True,
    box_on=False
)
```

[![fig2](/assets/classification_uncertain_best.png)](/assets/classification_uncertain_best.png)

## Plot overall of uncertainty

Plotting an overall of uncertainty estimated for the models.

```python
sorted_models = results.sort_values("accuracy", ascending=False).index.tolist()

pyramid = []
i = 0
for row in range(1, len(sorted_models)):
    if i + row <= len(sorted_models):
        pyramid.append(sorted_models[i:i+row])
        i += row
    else:
        break

if i < len(sorted_models):
    pyramid.append(sorted_models[i:])

if len(pyramid[-1]) < len(pyramid[-2]):
    pyramid[-2].extend(pyramid[-1])
    pyramid = pyramid[:-1]
```

```python
up = UncertaintyPlots()
```

```python
fig_un, ax_un = up.uncertainty(
    model_names=pyramid,
    probs=probas,
    X=grid_search.X_train,
    figsize=(20, 10),
    show_inline=True,
    cmap='RdYlGn',
    box_on=False
)
```

[![fig3](/assets/classification_uncertain_pyramid.png)](/assets/classification_uncertain_pyramid.png)

## Aleatory uncertainty and Epistemic uncertainty

```python
data_probas = pd.DataFrame(probas)
random: pd.Series = data_probas.mean(axis=1)
epistemic: pd.Series = data_probas.var(axis=1)
```

```python
up = UncertaintyPlots()
```

```python
fig_both, ax_both = up.uncertainty(
    model_names=[["Random uncertainty", "Epistemic uncertainty"]],
    probs={
        "Random uncertainty": random,
        "Epistemic uncertainty": epistemic
    },
    X=grid_search.X_train,
    figsize=(20, 6),
    cmap='RdYlGn',
    show_inline=True,
    box_on=False
)
```

[![fig4](/assets/classification_uncertain_aleatory_epistemic.png)](/assets/classification_uncertain_aleatory_epistemic.png)
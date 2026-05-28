# MLOPs-ModelVersioningAndRegistry
MLflow related work
### Without Autolog

``` python 
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")            # Also you can set cloud URI path Mention in mlflow.md
mlflow.set_experiment("Wine-Quality")

alpha=0.9
l1_ratio = 0.9

with mlflow.start_run():

    mlflow.set_tag("developer", "Dhaval Antala")
    mlflow.set_tag("Model", "elastic-net")

    mlflow.log_param('alpha', alpha)
    mlflow.log_param('l1_ratio', l1_ratio)

    lr = ElasticNet(alpha=alpha, l1_ratio=alpha)
    lr.fit(train_x,train_y)

    pred = lr.predict(test_x)

    rmse,mae,r2 = evaluate(test_y,pred)

    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mae', mae)
    mlflow.log_metric('r2', r2)

    mlflow.sklearn.log_model(lr, 'ElasticNet')

    print(f"Elastic net Params: alpha: {alpha}, l1_ratio: {l1_ratio}")
    print(f"Elastic net metric: rmse:{rmse}, mae:{mae},r2:{r2}")
```

### With Autolog
```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")            # Also you can set cloud URI path Mention in mlflow.md
mlflow.set_experiment("Wine-Quality")

mlflow.autolog()

with mlflow.start_run():

    lr = ElasticNet(alpha=alpha, l1_ratio=alpha)
    lr.fit(train_x,train_y)

    pred = lr.predict(test_x)

    rmse,mae,r2 = evaluate(test_y,pred)

    print(f"Elastic net Params: alpha: {alpha}, l1_ratio: {l1_ratio}")
    print(f"Elastic net metric: rmse:{rmse}, mae:{mae},r2:{r2}")
```

## HYPERPARAMETER TUNING AND TRACKING USING MLFLOW## Hyperparameters tuning

```python 
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
```

```python
# For child run tracking
mlflow.sklearn.autolog(max_tuning_runs=None)
```

```Python 
# Define parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
}

scoring_metrics = {
    "Accuracy": make_scorer(accuracy_score),
    "Precision": make_scorer(precision_score, average="macro", zero_division=0),
    "Recall": make_scorer(recall_score, average="macro"),
    "F1_Score": make_scorer(f1_score, average="macro")
} 
```

```python
with mlflow.start_run():
    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring=scoring_metrics,
        refit="Accuracy", 
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("test_f1_macro", f1_score(y_test, y_pred, average="macro"))

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV accuracy score: {grid_search.best_score_:.3f}")
```

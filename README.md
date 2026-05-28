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


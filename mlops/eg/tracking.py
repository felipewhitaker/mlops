import os
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

if __name__ == "__main__":

    mlflow.set_experiment(experiment_name = os.path.basename(__file__))

    data = pd.read_csv("data/winequality-red.csv")
    X, y = data.drop(columns="quality"), data["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0
    )

    pipe = make_pipeline(
        StandardScaler(), 
        LinearRegression()
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    r2 = r2_score(y_test, preds)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(pipe, "model")

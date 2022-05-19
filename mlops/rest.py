import os
import pickle
from datetime import datetime

from flask import Flask, request

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch_eg import Net

app = Flask(__name__)
# TODO allow other name for models folder
MODELS_FOLDER = "models"

@app.route("/train", methods = ["GET"])
def train():

    # TODO receive data file
    # e.g. data_file = request.files.get("data")
    data = pd.read_csv("data/winequality-red.csv")
    app.logger.debug(f"Loaded data")

    # TODO receive train test split qtd
    X, y = data.drop(columns = "quality"), data["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state = 42)
    app.logger.debug(f"{X_train.shape=} {X_test.shape=}, {y_train.shape=}, {y_test.shape=}")

    # TODO receive model name
    # e.g. model_name = request.args.get("model")
    # TODO create train and evaluate abstraction
    # TODO allow classification
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    app.logger.debug(f"Finished training")

    # TODO abstract model saving
    global MODELS_FOLDER
    os.makedirs(MODELS_FOLDER, exist_ok = True)
    # TODO better unique model naming
    # TODO store which features were used for the model to make sure prediction is done correctly
    model_path = os.path.join(MODELS_FOLDER, f"{clf.__class__.__name__}_{datetime.now():%Y%m%d%H%M}.pickle")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    app.logger.debug(f"Saved {model_path=}")

    preds = clf.predict(X_test)

    # precision, recall, fscore, support = precision_recall_fscore_support(y_test, preds)
    # app.logger.debug(f"Evaluated {precision=} {recall=} {fscore=} {support=}")
    return "{\"r2\":%.2f}"%r2_score(y_test, preds)

@app.route("/predict", methods = ["POST"])
def predict():

    # TODO receive which model to be used
    # TODO batch prediction

    # TODO abstract model loading
    global MODELS_FOLDER
    # FIXME consider other models
    model_file = sorted(
        (file for file in os.listdir(MODELS_FOLDER) if file.startswith("LinearRegression")),
        key = lambda file: os.path.getmtime(
            os.path.join(MODELS_FOLDER, file)
            ), 
        reverse = True
    )[0]
    app.logger.debug(f"Loading {model_file}")
    with open(os.path.join(MODELS_FOLDER, model_file), "rb") as f:
        clf = pickle.load(f)

    # must receive "Content-Type: application/json", otherwise `force=True`
    data = request.get_json() 

    # FIXME check data that is coming in (related to data that was used to train the model)
    # {
    #     "fixed acidity": 7.4,
    #     "volatile acidity": 0.7,
    #     "citric acid": 0.0,
    #     "residual sugar": 1.9,
    #     "chlorides": 0.076,
    #     "free sulfur dioxide": 11.0,
    #     "total sulfur dioxide": 34.0,
    #     "density": 0.9978,
    #     "pH": 3.51,
    #     "sulphates": 0.56,
    #     "alcohol": 9.4
    # }

    # FIXME is it needed to transform dict into DataFrame?
    preds = clf.predict(pd.DataFrame(data, index = [0]))

    return "{\"quality\":%.2f}"%preds[0]

@app.route("/predict_net", methods = ["POST"])
def predict_net():
    file_path = "models/dense"
    clf = Net.load(file_path)

    data = request.get_json()
    return "{\"quality\":%.2f}"%clf.predict_one(list(data.values()))

if __name__ == "__main__":
    # FIXME debug=True for development
    app.run(port = 8000, debug = True)
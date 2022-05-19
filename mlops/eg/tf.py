raise ImportError(
    "Depends on C++ DLLs - see https://support.microsoft.com/help/2977003/the-latest-supported-visual-c-downloads"
)

if __name__ == "__main__":

    import pandas as pd
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline

    dataset = pd.read_csv(
        "https://raw.githubusercontent.com/futurexskill/ml-model-deployment/main/storepurchasedata_large.csv"
    )
    X, y = dataset.drop(columns=["Purchased"]), dataset["Purchased"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(5, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # FIXME does tf models work on a pipeline?
    pipe = make_pipeline(StandardScaler(), model)

    pipe.fit(X_train, y_train, epochs=50)

    # FIXME is tf compatible with pipe.score?
    pipe.score(X_test, y_test)

    loss, accuracy = model.evaluate(X_test, y_test)

    print(model.summary())

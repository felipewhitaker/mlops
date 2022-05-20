# import nltk
# nltk.download("all")

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


if __name__ == "__main__":

    dataset = pd.read_csv(
        "https://raw.githubusercontent.com/futurexskill/ml-model-deployment/main/Restaurant_Reviews.tsv.txt",
        delimiter="\t",
        quoting=3,
    )

    # # this cleaning part is being done inside TfIDf
    # dataset["clean"] = dataset["Review"].apply(
    #     lambda text: " ".join(
    #         (ps.stem(word) for word in text.lower().split() if word not in stop)
    #     )
    # )

    pipe = make_pipeline(
        TfidfVectorizer(
            lowercase=True,
            # preprocessor=PorterStemmer().stem,
            stop_words=set(stopwords.words("english")),
            max_features=1000,
            min_df=3,
            max_df=0.6,
        ),
        KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2),
    )

    X, y = dataset["Review"], dataset["Liked"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    print(classification_report(y_test, preds))

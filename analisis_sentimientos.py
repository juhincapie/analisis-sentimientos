# Importamos librerías
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix


# 1
def read_data(input, columns_data):
    df = pd.read_csv(
        input,
        sep="\t",
        header=None,
        names=columns_data
    )
    return df

# 2


def group_data_for_label(df):
    df_tagged = df[(df["etiqueta"] >= 0)]
    df_untagged = df[(df["etiqueta"].isna())]

    x_tagged = df_tagged["mensaje"]
    y_tagged = df_tagged["etiqueta"]

    x_untagged = df_untagged["mensaje"]
    y_untagged = df_untagged["etiqueta"]

    return x_tagged, y_tagged, x_untagged, y_untagged

# 3


def data_set_preparation(x_tagged, y_tagged):

    x_train, x_test, y_train, y_test = train_test_split(
        x_tagged,
        y_tagged,
        test_size=0.1,
        random_state=12345
    )
    return x_train, x_test, y_train, y_test

# 4


def building_word_analyzer():

    stemmer = PorterStemmer()
    # ________________________________
    vectorizer = CountVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        lowercase=True
    )
    # _________________________________
    analyzer = vectorizer.build_analyzer()

    return lambda x: (stemmer.stem(w) for w in analyzer(x))

# 5


def instance_countvectorizer(analyzer):
    countVectorizer = CountVectorizer(
        analyzer=analyzer,
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        binary=True,
        max_df=1.0,
        min_df=5
    )
    return countVectorizer

# 6


def create_pipeline(countVectorizer):
    pipeline = Pipeline(
        steps=[
            ("countVectorizer", countVectorizer),
            ("bernoulli", BernoulliNB()),
        ],
    )
    return pipeline

# 7 y 8


def create_instance_gridsearchCV(pipeline, x_train, y_train):
    # 7__________________________________
    param_grid = {
        "bernoulli__alpha": np.arange(0.1, 1.01, 0.1),
    }
    # 8_________________________________
    gridSearchCV = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        refit=True,
        return_train_score=True,
    )
    # 9________________________________
    gridSearchCV.fit(x_train, y_train)

    return gridSearchCV

# 10


def assess_model(gridSearchCV, x_train, x_test, y_train, y_test):

    # trein____________________________________________
    cfm_train = confusion_matrix(
        y_true=y_train,
        y_pred=gridSearchCV.predict(x_train),
    )

    # test______________________________________________
    cfm_test = confusion_matrix(
        y_true=y_test,
        y_pred=gridSearchCV.predict(x_test),
    )
    return cfm_train, cfm_test

# 11


def predict_tags(gridSearchCV, x_untagged):

    y_untagged_pred = gridSearchCV.predict(x_untagged)

    return y_untagged_pred


def execute_sentiment_analysis():
    input = "data.txt"
    columns_data = ["mensaje", "etiqueta"]
    data = read_data(input, columns_data)
    x_tagged, y_tagged, x_untagged, y_untagged = group_data_for_label(data)
    x_train, x_test, y_train, y_test = data_set_preparation(x_tagged, y_tagged)
    analyzer = building_word_analyzer()
    countVectorizer = instance_countvectorizer(analyzer)
    pipeline = create_pipeline(countVectorizer)
    gridSearchCV = create_instance_gridsearchCV(pipeline, x_train, y_train)
    cfm_train, cfm_test = assess_model(
        gridSearchCV, x_train, x_test, y_train, y_test)
    y_untagged_pred = predict_tags(gridSearchCV, x_untagged)


if __name__ == '__main__':

    execute_sentiment_analysis()

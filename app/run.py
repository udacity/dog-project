import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table("Message", engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    cat_names = [
        "related",
        "request",
        "offer",
        "aid_related",
        "medical_help",
        "medical_products",
        "search_and_rescue",
        "security",
        "military",
        "child_alone",
        "water",
        "food",
        "shelter",
        "clothing",
        "money",
        "missing_people",
        "refugees",
        "death",
        "other_aid",
        "infrastructure_related",
        "transport",
        "buildings",
        "electricity",
        "tools",
        "hospitals",
        "shops",
        "aid_centers",
        "other_infrastructure",
        "weather_related",
        "floods",
        "storm",
        "fire",
        "earthquake",
        "cold",
        "other_weather",
        "direct_report",
    ]
    cat_counts = df[cat_names].sum().sort_values(ascending=False)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            "data": [Bar(x=genre_names, y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        },
        {
            "data": [Bar(x=cat_counts.index, y=cat_counts)],
            "layout": {
                "title": "Distribution of Message Categories",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Category"},
            },
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_probabilties = []
    for p in model.predict_proba([query]):
        try:
            p1 = p[0][1]
        except:
            p1 = 0.0
        classification_probabilties.append(p1)
    classification_results = dict(zip(df.columns[4:], classification_probabilties))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()

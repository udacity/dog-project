import os
from flask import Flask
from flask import render_template, request, jsonify


app = Flask(__name__)


images = os.listdir("static")
# print(images)


@app.route("/")
@app.route("/index")
def index():
    # # save user input in query
    selection = request.args.get("selection", "")
    return render_template("master.html", images=images, selection=selection)


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()

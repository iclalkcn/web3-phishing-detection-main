from flask import Flask, render_template, request
from utils import (
    get_prediction,
    get_lime_explanation,
    load_model_and_tokenizer_for_app,
)

# Initialize Flask app
app = Flask(__name__)

# Load hybrid model system
classifier, tokenizer, bert_model = load_model_and_tokenizer_for_app()


@app.route("/", methods=["GET", "POST"])
def main():
    try:
        if request.method == "POST":

            text = request.form
            messages = text["input"]

            print("INPUT:", messages)

            # Prediction
            full_prediction = get_prediction(
                classifier,
                tokenizer,
                bert_model,
                messages
            )

            # LIME explanation
            explanation = get_lime_explanation(
                classifier,
                tokenizer,
                bert_model,
                messages
            )

            if full_prediction is not None:
                label = full_prediction["LABEL"]
                probability = full_prediction["probability"]
            else:
                label = None
                probability = None

            print(full_prediction)
            print(explanation)

            return render_template(
                "show.html",
                label=label,
                probability=probability,
                explanation=explanation
            )

        else:
            return render_template("index.html")

    except Exception as e:
        print(f"An error occurred: {e}")
        return render_template("error.html", error=str(e))


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", debug=True)
    except Exception as e:
        print(f"An error occurred when starting the server: {e}")
        
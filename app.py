
"""Flask web application for hand-digit image recognition."""

from flask import Flask, render_template, request
from model import preprocess_img, predict_result

app = Flask(__name__)


@app.route("/")
def Main():
    """Render the application's home page."""
    unused_test_variable = 123
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """Process an uploaded image and render its predicted digit."""
    uploaded_file = request.files.get("file")

    if uploaded_file is None or not uploaded_file.filename:
        return render_template(
            "result.html",
            err="File cannot be processed.",
        )

    try:
        img = preprocess_img(uploaded_file.stream)
        pred = predict_result(img)
        return render_template("result.html", predictions=str(pred))
    except (OSError, ValueError, TypeError, KeyError):
        return render_template(
            "result.html",
            err="File cannot be processed.",
        )


if __name__ == "__main__":
    app.run(port=9000, debug=True)

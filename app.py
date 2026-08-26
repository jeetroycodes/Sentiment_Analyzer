from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# MODEL PATH
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "imdb_sentiment_model"
)


# =========================================================
# LOAD TENSORFLOW SAVEDMODEL
# =========================================================

print("========================================")
print("Loading TensorFlow SavedModel...")
print("========================================")

model = tf.saved_model.load(MODEL_PATH)

print("Model loaded successfully!")

print("Available signatures:")
print(model.signatures.keys())


# =========================================================
# GET SERVING FUNCTION
# =========================================================

if "serve" in model.signatures:
    serving_fn = model.signatures["serve"]

elif "serving_default" in model.signatures:
    serving_fn = model.signatures["serving_default"]

else:
    signature_name = list(model.signatures.keys())[0]
    serving_fn = model.signatures[signature_name]


print("\nInput signature:")
print(serving_fn.structured_input_signature)

print("\nOutput signature:")
print(serving_fn.structured_outputs)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_sentiment(review):

    # Convert review to TensorFlow string tensor
    review_tensor = tf.constant(
        [review],
        dtype=tf.string
    )

    # Call the TensorFlow model
    result = serving_fn(
        keras_tensor=review_tensor
    )

    # Get model output
    output_name = list(result.keys())[0]

    prediction = result[output_name]

    # Convert TensorFlow tensor to NumPy
    prediction = prediction.numpy()

    # Get probability
    probability = float(
        np.asarray(prediction).flatten()[0]
    )

    # =====================================================
    # SENTIMENT
    # =====================================================

    if probability >= 0.5:
        sentiment = "Positive"
        confidence = probability
    else:
        sentiment = "Negative"
        confidence = 1 - probability

    return sentiment, confidence


# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    sentiment = None
    confidence = None
    review = ""
    error = None

    if request.method == "POST":

        review = request.form.get(
            "review",
            ""
        ).strip()

        # Check empty input
        if not review:

            error = "Please enter a movie review."

        else:

            try:

                sentiment, confidence = predict_sentiment(
                    review
                )

                # Convert to percentage
                confidence = round(
                    confidence * 100,
                    2
                )

            except Exception as e:

                print("\nPrediction error:")
                print(e)

                error = (
                    "An error occurred while "
                    "analyzing the review."
                )

    return render_template(
        "index.html",
        sentiment=sentiment,
        confidence=confidence,
        review=review,
        error=error
    )


# =========================================================
# START FLASK SERVER
# =========================================================

if __name__ == "__main__":

    print("\n========================================")
    print("Starting Flask application...")
    print("========================================")

    port = int(
        os.environ.get("PORT", 5000)
    )

    print("Open your browser:")
    print(f"http://127.0.0.1:{port}")

    print("========================================\n")

    app.run(
        host="127.0.0.1",
        port=port,
        debug=True
    )
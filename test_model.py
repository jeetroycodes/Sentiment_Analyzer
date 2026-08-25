import tensorflow as tf

MODEL_PATH = "model/imdb_sentiment_model"

print("Loading model...")

model = tf.saved_model.load(MODEL_PATH)

print("Model loaded successfully!")

print("\nAvailable signatures:")
print(model.signatures.keys())

serving_fn = model.signatures["serve"]

print("\nInput signature:")
print(serving_fn.structured_input_signature)

print("\nOutput signature:")
print(serving_fn.structured_outputs)

# Your model expects the input name "keras_tensor"
review = tf.constant(
    ["This movie was absolutely fantastic and I loved it!"],
    dtype=tf.string
)

result = serving_fn(keras_tensor=review)

print("\nPrediction result:")
print(result)

# Get output
output_name = list(result.keys())[0]
prediction = result[output_name]

print("\nOutput name:")
print(output_name)

print("\nPrediction:")
print(prediction.numpy())
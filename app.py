from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# For login session management
app.secret_key = "your_secret_key"  # Replace with a secure secret in production

# Load your model
model = tf.keras.models.load_model("plant_disease_model_final.h5")
class_names = ["Healthy", "Powdery", "Rust"]

def preprocess_image(image, target_size=(128, 128)):
    img = image.convert("RGB").resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Landing page
@app.route("/")
def landing():
    return render_template("landing.html")

# Login page (GET) and handler (POST)
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    username = request.form.get("username")
    password = request.form.get("password")
    # Dummy credentials for demonstration
    if username == "user" and password == "password":
        session['logged_in'] = True
        return redirect(url_for("home"))
    else:
        return render_template("login.html", error="Invalid credentials. Please try again.")

# Main disease detection page (requires login)
@app.route("/home")
def home():
    if not session.get('logged_in'):
        return redirect(url_for("login"))
    return render_template("index.html")

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    img_processed = preprocess_image(img)
    preds = model.predict(img_processed)[0]
    pred_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    prediction = class_names[pred_idx]
    return jsonify({"prediction": prediction, "confidence": confidence})

# Logout route (optional)
@app.route("/logout")
def logout():
    session.pop('logged_in', None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)

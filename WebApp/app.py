from flask import Flask, render_template, request, redirect, session
from tinydb import TinyDB, Query
import os
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import pandas as pd

app = Flask(__name__)
app.secret_key = "yasodhas"


db = TinyDB("users.json")
users = db.table("users")


PRED_CSV = r"C:\Users\aalla\Downloads\PycharmProjects\PycharmProjects\human\.venv\New_ImageHumanActivityRecog_2026\Reports\test_predictions.csv"

pred_df = pd.read_csv(PRED_CSV)
pred_df["filename"] = pred_df["filename"].astype(str).str.strip().str.lower()

file_to_label = dict(zip(pred_df["filename"], pred_df["predicted_label"]))
MODEL_PATH = r"C:\Users\aalla\Downloads\PycharmProjects\PycharmProjects\human\.venv\New_ImageHumanActivityRecog_2026\ModelFiles\har_efficientnet_model.h5"
TRAIN_CSV = r"C:\Users\aalla\Downloads\PycharmProjects\PycharmProjects\human\.venv\New_ImageHumanActivityRecog_2026\Dataset\Training_set.csv"

model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224


train_df = pd.read_csv(TRAIN_CSV)
train_df["label"] = train_df["label"].astype(str)

CLASSES = sorted(train_df["label"].unique())
idx2label = {i: c for i, c in enumerate(CLASSES)}


UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



@app.route("/")
def home():
    return redirect("/login")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        username = request.form["username"]
        password = request.form["password"]
        email = request.form["email"]
        mobile = request.form["mobile"]

        # validations
        if len(password) < 5:
            return "Password must be at least 5 characters"

        if users.search(Query().username == username):
            return "Username already exists"

        users.insert({
            "name": name,
            "username": username,
            "password": password,
            "email": email,
            "mobile": mobile
        })

        return redirect("/login")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = users.search(
            (Query().username == username) &
            (Query().password == password)
        )

        if user:
            session["user"] = username
            return redirect("/dashboard")
        else:
            return "Invalid credentials"

    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login")

    return render_template("dashboard.html")


@app.route("/train")
def train():
    if "user" not in session:
        return redirect("/login")

    return render_template("train.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect("/login")

    result = None
    img_path = None

    if request.method == "POST":
        file = request.files["image"]
        filename = secure_filename(file.filename)

        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # -------- USE CSV FOR CORRECT LABEL -------- #
        label = file_to_label.get(filename.lower().strip())

        if label:
            result = f"Prediction: {label}"
        else:
            # fallback to model if not found in CSV
            img = Image.open(filepath).resize((IMG_SIZE, IMG_SIZE))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            preds = model.predict(img)
            pred_class = int(np.argmax(preds))

            result = f"Predicted Class Index: {pred_class}"

        img_path = filepath

    return render_template("predict.html", result=result, img_path=img_path)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")


if __name__ == "__main__":
    app.run(debug=True)
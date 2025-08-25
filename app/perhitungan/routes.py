import os
import traceback
import numpy as np
import pandas as pd
import pickle
from flask import Blueprint, request, jsonify, render_template, send_file
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

perhitungan_bp = Blueprint(
    "perhitungan",
    __name__,
    url_prefix="/perhitungan",
    template_folder="templates",
    static_folder="static",
)

# Folder utama
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT_UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "perhitungan")
os.makedirs(ROOT_UPLOAD_FOLDER, exist_ok=True)

# Jalur model & encoder
MODEL_PATH = os.path.join(ROOT_UPLOAD_FOLDER, "model_mlp_custom.keras")
LABEL_ENCODER_PATH = os.path.join(ROOT_UPLOAD_FOLDER, "label_encoder.pkl")

model = None
le = None


def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if "Status" not in df.columns:
        raise ValueError("Kolom 'Status' tidak ditemukan di CSV.")

    # Fitur (TF-IDF hasil preprocessing sebelumnya)
    X = df.drop(columns=["Status"]).to_numpy()

    # Label encoding
    y_raw = df["Status"].astype(str).to_numpy()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    if len(np.unique(y_encoded)) < 2:
        raise ValueError("Data harus memiliki minimal 2 kelas berbeda.")

    # Simpan encoder otomatis
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    return X, y_encoded, label_encoder, len(np.unique(y_encoded)), df


def build_and_train_model(X, y, output_dim):
    model = Sequential(
        [
            Input(shape=(X.shape[1],)),
            Dense(64, activation="relu", kernel_initializer=RandomUniform(-0.12, 0.12)),
            Dense(32, activation="relu"),
            Dense(output_dim, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=["accuracy"]
    )
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # Simpan model otomatis
    model.save(MODEL_PATH)
    return model


@perhitungan_bp.route("/")
def index():
    return render_template("perhitungan.html", page_name="perhitungan")


@perhitungan_bp.route("/process", methods=["POST"])
def process_csv():
    global model, le
    file = request.files.get("file")
    if not file or not file.filename or not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "File harus CSV."}), 400

    try:
        csv_path = os.path.join(ROOT_UPLOAD_FOLDER, file.filename)
        file.save(csv_path)

        X, y, encoder, output_dim, df_clean = load_and_prepare_data(csv_path)

        model = build_and_train_model(X, y, output_dim)
        le = encoder

        # Prediksi training set (hanya evaluasi internal)
        y_pred = le.inverse_transform(np.argmax(model.predict(X), axis=1))
        y_actual = le.inverse_transform(y)

        df_clean["Prediksi"] = y_pred

        # Hitung metrik evaluasi
        acc = accuracy_score(y_actual, y_pred)
        prec = precision_score(y_actual, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_actual, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_actual, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_actual, y_pred, labels=np.unique(y_actual)).tolist()
        labels = list(np.unique(y_actual))

        return jsonify(
            {
                "message": "Pelatihan selesai.",
                "train": df_clean.to_dict(orient="records"),
                "accuracy": round(float(acc) * 100, 2),
                "precision": round(float(prec) * 100, 2),
                "recall": round(float(rec) * 100, 2),
                "f1": round(float(f1) * 100, 2),
                "labels": labels,
                "confusion": cm,
            }
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses: {str(e)}"}), 500


@perhitungan_bp.route("/download/<filename>")
def download_file(filename):
    safe_files = {
        "model_mlp_custom.keras": MODEL_PATH,
        "label_encoder.pkl": LABEL_ENCODER_PATH,
    }
    path = safe_files.get(filename)
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({"error": f"File '{filename}' tidak ditemukan."}), 404


@perhitungan_bp.route("/clear", methods=["POST"])
def clear_data():
    global model, le
    for path in [MODEL_PATH, LABEL_ENCODER_PATH]:
        if os.path.exists(path):
            os.remove(path)
    model, le = None, None
    return jsonify({"message": "Semua data dan model berhasil dihapus."})


@perhitungan_bp.route("/copy-download-model")
def copy_and_download_model():
    try:
        return send_file(MODEL_PATH, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"Gagal mendownload model: {str(e)}"}), 500


@perhitungan_bp.route("/copy-download-encoder")
def copy_and_download_encoder():
    try:
        return send_file(LABEL_ENCODER_PATH, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"Gagal mendownload encoder: {str(e)}"}), 500

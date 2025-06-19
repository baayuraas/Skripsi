import os
import shutil
import traceback
import numpy as np
import pandas as pd
import pickle
from flask import Blueprint, request, jsonify, render_template, send_file
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder internal (relatif terhadap modul perhitungan)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "perhitungan")
EVAL_UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "evaluasi")

# Folder global target di luar app: pakkur/uploads/perhitungan
ROOT_UPLOAD_FOLDER = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "uploads", "perhitungan")
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EVAL_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ROOT_UPLOAD_FOLDER, exist_ok=True)


CSV_PATH = os.path.join(UPLOAD_FOLDER, "train.csv")
EVAL_PATH = os.path.join(UPLOAD_FOLDER, "data_uji.csv")
LATIH_PATH = os.path.join(UPLOAD_FOLDER, "data_latih.csv")
MODEL_PATH = os.path.join(UPLOAD_FOLDER, "model_mlp_custom.keras")
LABEL_ENCODER_PATH = os.path.join(UPLOAD_FOLDER, "label_encoder.pkl")

model = None
le = None


def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "Status" not in df.columns:
        raise ValueError("Kolom 'Status' tidak ditemukan.")

    df_features = df.drop(columns=["Status"], errors="ignore").apply(
        pd.to_numeric, errors="coerce"
    )
    valid_mask = df_features.notna().all(axis=1)
    df_clean = df[valid_mask].copy()

    if df_clean.empty:
        raise ValueError("Data tidak valid. Pastikan semua fitur numerik.")

    X = df_clean.drop(columns=["Status"]).to_numpy()
    y_raw = df_clean["Status"].astype(str).to_numpy()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    if len(np.unique(y_encoded)) < 2:
        raise ValueError("Data harus memiliki minimal 2 kelas berbeda.")

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)
    with open(os.path.join(EVAL_UPLOAD_FOLDER, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    with open(os.path.join(ROOT_UPLOAD_FOLDER, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    return X, y_encoded, label_encoder, len(np.unique(y_encoded)), df_clean


def build_and_train_model(X, y, output_dim):
    model = Sequential(
        [
            Dense(
                64,
                activation="relu",
                kernel_initializer=RandomUniform(-0.12, 0.12),
                input_shape=(X.shape[1],),
            ),
            Dense(32, activation="relu"),
            Dense(output_dim, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=["accuracy"]
    )
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    model.save(MODEL_PATH)
    model.save(os.path.join(EVAL_UPLOAD_FOLDER, "model_mlp_custom.keras"))
    model.save(os.path.join(ROOT_UPLOAD_FOLDER, "model_mlp_custom.keras"))
    return model


@perhitungan_bp.route("/")
def index():
    return render_template("perhitungan.html")


@perhitungan_bp.route("/process", methods=["POST"])
def process_csv():
    global model, le
    file = request.files.get("file")
    if not file or not file.filename or not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "File harus CSV."}), 400

    try:
        file.save(CSV_PATH)
        X, y, encoder, output_dim, df_clean = load_and_prepare_data(CSV_PATH)

        X_train, X_eval, y_train, y_eval, df_train, df_eval = train_test_split(
            X, y, df_clean, test_size=0.3, random_state=42, stratify=y
        )

        model = build_and_train_model(X_train, y_train, output_dim)
        le = encoder

        y_pred_train = le.inverse_transform(np.argmax(model.predict(X_train), axis=1))
        y_actual_train = le.inverse_transform(y_train)

        df_train["Asli"] = y_actual_train
        df_train["Prediksi"] = y_pred_train
        df_train.to_csv(LATIH_PATH, index=False)
        df_eval.to_csv(EVAL_PATH, index=False)

        acc = accuracy_score(y_actual_train, y_pred_train)
        prec = precision_score(
            y_actual_train, y_pred_train, average="weighted", zero_division=0
        )
        rec = recall_score(
            y_actual_train, y_pred_train, average="weighted", zero_division=0
        )
        f1 = f1_score(y_actual_train, y_pred_train, average="weighted", zero_division=0)
        cm = confusion_matrix(
            y_actual_train, y_pred_train, labels=np.unique(y_actual_train)
        ).tolist()
        labels = list(np.unique(y_actual_train))

        return jsonify(
            {
                "message": "✅ Pelatihan selesai.",
                "train": df_train.to_dict(orient="records"),
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
        "train.csv": CSV_PATH,
        "model_mlp_custom.keras": MODEL_PATH,
        "label_encoder.pkl": LABEL_ENCODER_PATH,
        "data_uji.csv": EVAL_PATH,
        "data_latih.csv": LATIH_PATH,
    }
    path = safe_files.get(filename)
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({"error": f"File '{filename}' tidak ditemukan."}), 404


@perhitungan_bp.route("/clear", methods=["POST"])
def clear_data():
    global model, le
    for path in [CSV_PATH, EVAL_PATH, LATIH_PATH, MODEL_PATH, LABEL_ENCODER_PATH]:
        if os.path.exists(path):
            os.remove(path)
    model, le = None, None
    return jsonify({"message": "✅ Semua data dan model berhasil dihapus."})


@perhitungan_bp.route("/copy-download-model")
def copy_and_download_model():
    try:
        shutil.copy2(
            MODEL_PATH, os.path.join(EVAL_UPLOAD_FOLDER, "model_mlp_custom.keras")
        )
        shutil.copy2(
            MODEL_PATH, os.path.join(ROOT_UPLOAD_FOLDER, "model_mlp_custom.keras")
        )
        return send_file(MODEL_PATH, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"Gagal mendownload model: {str(e)}"}), 500


@perhitungan_bp.route("/copy-download-encoder")
def copy_and_download_encoder():
    try:
        shutil.copy2(
            LABEL_ENCODER_PATH, os.path.join(EVAL_UPLOAD_FOLDER, "label_encoder.pkl")
        )
        shutil.copy2(
            LABEL_ENCODER_PATH, os.path.join(ROOT_UPLOAD_FOLDER, "label_encoder.pkl")
        )
        return send_file(LABEL_ENCODER_PATH, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"Gagal mendownload encoder: {str(e)}"}), 500

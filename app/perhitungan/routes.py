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
from tensorflow.keras.models import Sequential, load_model
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
HASIL_CSV_PATH = os.path.join(
    ROOT_UPLOAD_FOLDER, "hasil_perhitungan.csv"
)  # File untuk menyimpan hasil

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


@perhitungan_bp.route("/check-result", methods=["GET"])
def check_result():
    """Memeriksa apakah ada hasil perhitungan yang tersimpan"""
    if (
        os.path.exists(HASIL_CSV_PATH)
        and os.path.exists(MODEL_PATH)
        and os.path.exists(LABEL_ENCODER_PATH)
    ):
        return jsonify({"has_result": True})
    else:
        return jsonify({"has_result": False})


@perhitungan_bp.route("/process", methods=["POST"])
def process_csv():
    global model, le

    # Pertama, cek apakah sudah ada hasil yang tersimpan
    if (
        os.path.exists(HASIL_CSV_PATH)
        and os.path.exists(MODEL_PATH)
        and os.path.exists(LABEL_ENCODER_PATH)
    ):
        try:
            # Langsung muat hasil yang tersimpan
            df = pd.read_csv(HASIL_CSV_PATH, encoding="utf-8-sig")

            # Muat model dan encoder
            model = load_model(MODEL_PATH)
            with open(LABEL_ENCODER_PATH, "rb") as f:
                le = pickle.load(f)

            # Hitung metrik evaluasi
            if "Status" in df.columns and "Prediksi" in df.columns:
                y_actual = df["Status"].astype(str)
                y_pred = df["Prediksi"].astype(str)

                acc = accuracy_score(y_actual, y_pred)
                prec = precision_score(
                    y_actual, y_pred, average="weighted", zero_division=0
                )
                rec = recall_score(
                    y_actual, y_pred, average="weighted", zero_division=0
                )
                f1 = f1_score(y_actual, y_pred, average="weighted", zero_division=0)
                cm = confusion_matrix(
                    y_actual, y_pred, labels=np.unique(y_actual)
                ).tolist()
                labels = list(np.unique(y_actual))

                return jsonify(
                    {
                        "message": "Data hasil sebelumnya berhasil dimuat.",
                        "train": df.to_dict(orient="records"),
                        "accuracy": round(float(acc) * 100, 2),
                        "precision": round(float(prec) * 100, 2),
                        "recall": round(float(rec) * 100, 2),
                        "f1": round(float(f1) * 100, 2),
                        "labels": labels,
                        "confusion": cm,
                    }
                )
            else:
                return jsonify(
                    {
                        "message": "Data hasil sebelumnya berhasil dimuat.",
                        "train": df.to_dict(orient="records"),
                    }
                )

        except Exception as e:
            # Jika ada error, lanjutkan dengan proses normal
            traceback.print_exc()
            # Jangan return error, biarkan lanjut ke proses normal

    # Jika tidak ada hasil tersimpan, lanjutkan dengan proses normal
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

        # Simpan hasil ke CSV
        df_clean.to_csv(HASIL_CSV_PATH, index=False, encoding="utf-8-sig")

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


@perhitungan_bp.route("/load-result", methods=["GET"])
def load_result():
    """Memuat hasil perhitungan yang tersimpan"""
    if not os.path.exists(HASIL_CSV_PATH):
        return jsonify({"error": "Tidak ada data hasil yang tersimpan."}), 404

    try:
        df = pd.read_csv(HASIL_CSV_PATH, encoding="utf-8-sig")

        # Hitung metrik evaluasi jika data memiliki kolom Status dan Prediksi
        if "Status" in df.columns and "Prediksi" in df.columns:
            y_actual = df["Status"].astype(str)
            y_pred = df["Prediksi"].astype(str)

            acc = accuracy_score(y_actual, y_pred)
            prec = precision_score(
                y_actual, y_pred, average="weighted", zero_division=0
            )
            rec = recall_score(y_actual, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_actual, y_pred, average="weighted", zero_division=0)
            cm = confusion_matrix(y_actual, y_pred, labels=np.unique(y_actual)).tolist()
            labels = list(np.unique(y_actual))

            return jsonify(
                {
                    "train": df.to_dict(orient="records"),
                    "accuracy": round(float(acc) * 100, 2),
                    "precision": round(float(prec) * 100, 2),
                    "recall": round(float(rec) * 100, 2),
                    "f1": round(float(f1) * 100, 2),
                    "labels": labels,
                    "confusion": cm,
                }
            )
        else:
            return jsonify(
                {
                    "train": df.to_dict(orient="records"),
                    "message": "Data hasil dimuat, tetapi tidak memiliki kolom yang diperlukan untuk evaluasi.",
                }
            )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memuat data hasil: {str(e)}"}), 500


@perhitungan_bp.route("/download/<filename>")
def download_file(filename):
    safe_files = {
        "model_mlp_custom.keras": MODEL_PATH,
        "label_encoder.pkl": LABEL_ENCODER_PATH,
        "hasil_perhitungan.csv": HASIL_CSV_PATH,
    }
    path = safe_files.get(filename)
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({"error": f"File '{filename}' tidak ditemukan."}), 404


@perhitungan_bp.route("/clear", methods=["POST"])
def clear_data():
    global model, le
    for path in [MODEL_PATH, LABEL_ENCODER_PATH, HASIL_CSV_PATH]:
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

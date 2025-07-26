import os
import logging
import pickle
import html
import numpy as np
from typing import Optional
from flask import Blueprint, request, jsonify, render_template
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from app.terjemahan.routes import translate_large_text
from app.preproses.routes import proses_baris_aman

pengujian_bp = Blueprint(
    "pengujian", __name__, url_prefix="/pengujian", template_folder="templates"
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_PATH = os.path.join(BASE_DIR, "uploads/perhitungan/model_mlp_custom.keras")
LABEL_PATH = os.path.join(BASE_DIR, "uploads/perhitungan/label_encoder.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "uploads/tfidf/tfidf.pkl")

ml_model = None
label_encoder: Optional[LabelEncoder] = None
tfidf_vectorizer: Optional[TfidfVectorizer] = None

# Load model & komponen
try:
    ml_model = load_model(MODEL_PATH)
    with open(LABEL_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    logging.info("Model, LabelEncoder, dan TF-IDF vectorizer berhasil dimuat.")
except Exception as e:
    logging.error(f"Gagal load model atau komponen: {e}")
    ml_model, label_encoder, tfidf_vectorizer = None, None, None


def ensure_model_ready():
    if not ml_model or not label_encoder or not tfidf_vectorizer:
        raise RuntimeError("Model atau komponen belum siap.")


@pengujian_bp.route("/", methods=["GET"])
def index():
    return render_template("data_pengujian.html", page_name="pengujian")


@pengujian_bp.route("/health", methods=["GET"])
def health_check():
    try:
        ensure_model_ready()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


@pengujian_bp.route("/predict", methods=["POST"])
def predict_sentimen():
    try:
        ensure_model_ready()

        data = request.get_json()
        if not data or "ulasan" not in data:
            return jsonify({"error": "Input tidak valid."}), 400

        ulasan = data["ulasan"].strip()
        if not ulasan:
            return jsonify({"error": "Ulasan kosong."}), 400

        # Escape HTML tags (prevent injection in render)
        ulasan = html.escape(ulasan)

        # Terjemahan jika perlu
        terjemahan = translate_large_text(ulasan, "id")
        is_translated = terjemahan.strip() != ulasan.strip()

        # Preprocessing
        hasil_prepro_list = proses_baris_aman(terjemahan)
        if not hasil_prepro_list or not hasil_prepro_list[-1]:
            return jsonify({"error": "Preprocessing gagal."}), 500

        hasil_prepro = hasil_prepro_list[-1]

        # TF-IDF transform
        vector = tfidf_vectorizer.transform([hasil_prepro])
        pred = ml_model.predict(
            vector.toarray()
        )  # model harus support sparse agar toarray bisa dihindari

        label_prediksi = label_encoder.inverse_transform([np.argmax(pred)])[0]

        # Evaluasi label
        label_asli = data.get("label")
        label_match = None

        if label_asli:
            label_asli = label_asli.strip()
            valid_labels = set(label.lower() for label in label_encoder.classes_)
            if label_asli.lower() not in valid_labels:
                return jsonify({"error": "Label asli tidak valid."}), 400

            label_match = label_prediksi.strip().lower() == label_asli.lower()

        return jsonify(
            {
                "terjemahan": terjemahan if is_translated else "",
                "preprocessing": hasil_prepro,
                "prediksi": label_prediksi,
                "label_asli": label_asli,
                "label_match": label_match,
            }
        )

    except Exception as e:
        logging.exception("Gagal memproses prediksi.")
        return jsonify({"error": f"Kesalahan server: {str(e)}"}), 500

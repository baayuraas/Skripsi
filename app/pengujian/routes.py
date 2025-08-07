import os
import logging
import pickle
import html
import numpy as np
from flask import Blueprint, request, jsonify, render_template
from keras.models import load_model

from app.terjemahan.routes import translate_large_text
from app.preproses.routes import proses_baris_aman

pengujian_bp = Blueprint(
    "pengujian", __name__, url_prefix="/pengujian", template_folder="templates"
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_PATH = os.path.join(BASE_DIR, "uploads", "perhitungan", "model_mlp_custom.keras")
LABEL_PATH = os.path.join(BASE_DIR, "uploads", "perhitungan", "label_encoder.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "uploads", "perhitungan", "tfidf.pkl")


def load_all_model_components():
    """Load model, label encoder, dan TF-IDF vectorizer dari file"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}. "
            "Silakan jalankan proses perhitungan terlebih dahulu."
        )
    if not os.path.exists(LABEL_PATH):
        raise FileNotFoundError(
            f"LabelEncoder tidak ditemukan: {LABEL_PATH}. "
            "Silakan jalankan proses perhitungan terlebih dahulu."
        )
    if not os.path.exists(TFIDF_PATH):
        raise FileNotFoundError(
            f"TF-IDF vectorizer tidak ditemukan: {TFIDF_PATH}. "
            "Silakan jalankan proses perhitungan terlebih dahulu."
        )

    model = load_model(MODEL_PATH)
    with open(LABEL_PATH, "rb") as f:
        encoder = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, encoder, vectorizer


@pengujian_bp.route("/", methods=["GET"])
def index():
    return render_template("data_pengujian.html", page_name="pengujian")


@pengujian_bp.route("/health", methods=["GET"])
def health_check():
    try:
        load_all_model_components()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


@pengujian_bp.route("/predict", methods=["POST"])
def predict_sentimen():
    try:
        # Load model & komponen setiap kali prediksi
        model, encoder, vectorizer = load_all_model_components()

        data = request.get_json()
        if not data or "ulasan" not in data:
            return jsonify({"error": "Input tidak valid."}), 400

        ulasan = data["ulasan"].strip()
        if not ulasan:
            return jsonify({"error": "Ulasan kosong."}), 400

        ulasan = html.escape(ulasan)
        terjemahan = translate_large_text(ulasan, "id")
        is_translated = terjemahan.strip() != ulasan.strip()

        hasil_prepro_list = proses_baris_aman(terjemahan)
        if not hasil_prepro_list or not hasil_prepro_list[-1]:
            return jsonify({"error": "Preprocessing gagal."}), 500

        hasil_prepro = hasil_prepro_list[-1]
        vector = vectorizer.transform([hasil_prepro])

        # Coba prediksi langsung sparse, kalau gagal pakai toarray
        try:
            pred = model.predict(vector)
        except Exception:
            pred = model.predict(vector.toarray())

        label_prediksi = encoder.inverse_transform([np.argmax(pred)])[0]

        label_asli = data.get("label")
        label_match = None

        if label_asli:
            label_asli = label_asli.strip()
            valid_labels = set(label.lower() for label in encoder.classes_)
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

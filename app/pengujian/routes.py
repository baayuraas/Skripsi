import os
import logging
import pickle
import html
import numpy as np
from functools import lru_cache
from typing import Optional
from pydantic import BaseModel, ValidationError, constr
from flask import Blueprint, request, jsonify, render_template
from keras.models import load_model
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Setup Blueprint
pengujian_bp = Blueprint(
    "pengujian", __name__, url_prefix="/pengujian", template_folder="templates"
)

# Konfigurasi Path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_PATH = os.path.join(BASE_DIR, "uploads", "perhitungan", "model_mlp_custom.keras")
LABEL_PATH = os.path.join(BASE_DIR, "uploads", "perhitungan", "label_encoder.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "uploads", "perhitungan", "tfidf.pkl")


# Model untuk Validasi Input
class PredictionRequest(BaseModel):
    ulasan: constr(min_length=1, max_length=1000)
    label: Optional[str] = None


# Cache Model
@lru_cache(maxsize=1)
def load_all_model_components():
    """Load model dengan caching"""
    try:
        if not all(
            os.path.exists(path) for path in [MODEL_PATH, LABEL_PATH, TFIDF_PATH]
        ):
            missing = [
                p for p in [MODEL_PATH, LABEL_PATH, TFIDF_PATH] if not os.path.exists(p)
            ]
            raise ModelLoadingError(f"File model tidak ditemukan: {', '.join(missing)}")

        model = load_model(MODEL_PATH)
        with open(LABEL_PATH, "rb") as f:
            encoder = pickle.load(f)
        with open(TFIDF_PATH, "rb") as f:
            vectorizer = pickle.load(f)

        return model, encoder, vectorizer
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise ModelLoadingError("Sistem prediksi sedang tidak tersedia")


# Routes
@pengujian_bp.route("/", methods=["GET"])
def index():
    return render_template("data_pengujian.html", page_name="pengujian")


@pengujian_bp.route("/health", methods=["GET"])
def health_check():
    try:
        load_all_model_components()
        return jsonify({"status": "ok"}), 200
    except ModelLoadingError as e:
        return jsonify({"status": "error", "detail": str(e)}), 503
    except Exception as e:
        return jsonify({"status": "error", "detail": "Kesalahan internal"}), 500


@pengujian_bp.route("/predict", methods=["POST"])
@limiter.limit("10 per minute")  # Rate limiting
def predict_sentimen():
    try:
        # Validasi Input
        try:
            data = PredictionRequest(**request.get_json())
        except ValidationError as e:
            return jsonify({"error": "Input tidak valid", "details": e.errors()}), 400

        # Load Model
        try:
            model, encoder, vectorizer = load_all_model_components()
        except ModelLoadingError as e:
            return jsonify({"error": str(e)}), 503

        # Proses Teks
        ulasan = html.escape(data.ulasan.strip())
        terjemahan = translate_large_text(ulasan, "id")
        is_translated = terjemahan.strip() != ulasan.strip()

        hasil_prepro = proses_baris_aman(terjemahan)
        if not hasil_prepro or not hasil_prepro[-1]:
            return jsonify({"error": "Gagal memproses teks"}), 400

        # Prediksi
        try:
            vector = vectorizer.transform([hasil_prepro[-1]])
            pred = model.predict(vector)
            label_prediksi = encoder.inverse_transform([np.argmax(pred)])[0]
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return jsonify({"error": "Gagal melakukan prediksi"}), 500

        # Validasi Label (jika ada)
        label_match = None
        if data.label:
            valid_labels = {label.lower() for label in encoder.classes_}
            if data.label.lower() not in valid_labels:
                return jsonify({"error": "Label tidak valid"}), 400
            label_match = label_prediksi.lower() == data.label.lower()

        return jsonify(
            {
                "hasil": {"prediksi": label_prediksi, "kecocokan_label": label_match},
                "proses": {
                    "terjemahan": terjemahan if is_translated else None,
                    "preprocessing": hasil_prepro[-1],
                },
            }
        )

    except Exception as e:
        logging.exception("Kesalahan tidak terduga")
        return jsonify({"error": "Terjadi kesalahan pada sistem"}), 500

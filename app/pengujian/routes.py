import os
import logging
import pickle
import html
import numpy as np
from functools import lru_cache
from typing import Optional
from pydantic import BaseModel, ValidationError, constr, Field
from flask import Blueprint, request, jsonify, render_template
from keras.models import load_model
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Limiter
limiter = Limiter(
    key_func=get_remote_address, default_limits=["200 per day", "50 per hour"]
)

# Blueprint
pengujian_bp = Blueprint(
    "pengujian", __name__, url_prefix="/pengujian", template_folder="templates"
)

# Path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_PATH = os.path.join(BASE_DIR, "uploads", "perhitungan", "model_mlp_custom.keras")
LABEL_PATH = os.path.join(BASE_DIR, "uploads", "perhitungan", "label_encoder.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "uploads", "perhitungan", "tfidf.pkl")


# Custom Exception
class ModelLoadingError(Exception):
    pass


ConstrainedStr = constr(min_length=1, max_length=1000)

class PredictionRequest(BaseModel):
    ulasan: str = Field(..., min_length=1, max_length=1000)
    label: Optional[str] = None


# Dummy translation & preprocessing
def translate_large_text(text: str, target_lang: str) -> str:
    return text  # TODO: hubungkan ke API translate


def proses_baris_aman(text: str) -> list:
    return [text.strip()]  # TODO: tambahkan preprocessing asli


@lru_cache(maxsize=1)
def load_all_model_components():
    if not os.path.exists(MODEL_PATH):
        raise ModelLoadingError(f"Model tidak ditemukan: {MODEL_PATH}")
    if not os.path.exists(LABEL_PATH):
        raise ModelLoadingError(f"LabelEncoder tidak ditemukan: {LABEL_PATH}")
    if not os.path.exists(TFIDF_PATH):
        raise ModelLoadingError(f"TF-IDF vectorizer tidak ditemukan: {TFIDF_PATH}")

    try:
        model = load_model(MODEL_PATH)
        with open(LABEL_PATH, "rb") as f:
            encoder = pickle.load(f)
        with open(TFIDF_PATH, "rb") as f:
            vectorizer = pickle.load(f)
    except Exception as e:
        raise ModelLoadingError(f"Gagal memuat model: {e}")

    return model, encoder, vectorizer


# ROUTES
@pengujian_bp.route("/", methods=["GET"])
def index():
    return render_template("data_pengujian.html", page_name="pengujian")


@pengujian_bp.route("/health", methods=["GET"])
def health_check():
    try:
        load_all_model_components()
        return jsonify({"status": "ok", "message": "Sistem berjalan normal"}), 200
    except ModelLoadingError as e:
        logger.error(f"Error loading model: {e}")
        return jsonify({
            "status": "error",
            "code": "MODEL_UNAVAILABLE",
            "solution": "Silakan jalankan proses training terlebih dahulu"
        }), 503
    except Exception as ex:
        logger.exception(f"Unexpected error: {ex}")
        return jsonify({
            "status": "error",
            "code": "SERVER_ERROR",
            "detail": "Internal server error"
        }), 500


@pengujian_bp.route("/predict", methods=["POST"])
@limiter.limit("10 per minute")
def predict_sentimen():
    try:
        try:
            data = PredictionRequest(**request.get_json())
        except ValidationError as e:
            return jsonify({"error": "Input tidak valid", "details": e.errors()}), 400

        try:
            model, encoder, vectorizer = load_all_model_components()
        except ModelLoadingError as e:
            return jsonify({"error": str(e)}), 503

        ulasan = html.escape(data.ulasan.strip())
        terjemahan = translate_large_text(ulasan, "id")
        is_translated = terjemahan.strip() != ulasan.strip()

        hasil_prepro = proses_baris_aman(terjemahan)
        if not hasil_prepro or not hasil_prepro[-1]:
            return jsonify({"error": "Preprocessing gagal"}), 400

        vector = vectorizer.transform([hasil_prepro[-1]])
        pred = model.predict(vector)
        label_prediksi = encoder.inverse_transform([np.argmax(pred)])[0]

        label_match = None
        if data.label:
            valid_labels = {label.lower() for label in encoder.classes_}
            if data.label.lower() not in valid_labels:
                return jsonify({"error": "Label tidak valid", "valid_labels": list(valid_labels)}), 400
            label_match = label_prediksi.lower() == data.label.lower()

        return jsonify({
            "result": {
                "prediction": label_prediksi,
                "label_match": label_match
            },
            "process": {
                "translated": is_translated,
                "translation_text": terjemahan if is_translated else ulasan,
                "preprocessed_text": hasil_prepro[-1],
            },
        })

    except Exception as e:
        logger.exception(f"Error predict: {e}")
        return jsonify({"error": "System error", "detail": str(e)}), 500

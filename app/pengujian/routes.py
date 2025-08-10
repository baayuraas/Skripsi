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


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Limiter
limiter = Limiter(
    key_func=get_remote_address, default_limits=["200 per day", "50 per hour"]
)

# Setup Blueprint
pengujian_bp = Blueprint(
    "pengujian", __name__, url_prefix="/pengujian", template_folder="templates"
)

# Path Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_PATH = os.path.join(BASE_DIR, "uploads", "perhitungan", "model_mlp_custom.keras")
LABEL_PATH = os.path.join(BASE_DIR, "uploads", "perhitungan", "label_encoder.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "uploads", "perhitungan", "tfidf.pkl")


# Custom Exceptions
class ModelLoadingError(Exception):
    pass


# Input Validation Model
class PredictionRequest(BaseModel):
    ulasan: constr(min_length=1, max_length=1000)
    label: Optional[str] = None


# Translation and Preprocessing Functions
def translate_large_text(text: str, target_lang: str) -> str:
    """Dummy translation function - replace with actual implementation"""
    return text  # In production, connect to translation API


def proses_baris_aman(text: str) -> list:
    """Dummy preprocessing function - replace with actual implementation"""
    return [text.strip()]  # In production, add actual preprocessing steps


# Cached Model Loading
@lru_cache(maxsize=1)
def load_all_model_components():
    """Load and cache model components"""
    try:
        # Check if all model files exist
        required_files = [MODEL_PATH, LABEL_PATH, TFIDF_PATH]
        if not all(os.path.exists(path) for path in required_files):
            missing = [p for p in required_files if not os.path.exists(p)]
            raise ModelLoadingError(f"Model files not found: {', '.join(missing)}")

        # Load components
        model = load_model(MODEL_PATH)
        with open(LABEL_PATH, "rb") as f:
            encoder = pickle.load(f)
        with open(TFIDF_PATH, "rb") as f:
            vectorizer = pickle.load(f)

        return model, encoder, vectorizer

    except Exception as e:
        logger.error(f"Model loading error: {str(e)}")
        raise ModelLoadingError("Prediction system is currently unavailable")


# Routes
@pengujian_bp.route("/", methods=["GET"])
def index():
    """Render testing page"""
    return render_template("data_pengujian.html", page_name="pengujian")


@pengujian_bp.route("/health", methods=["GET"])
def health_check():
    """Endpoint untuk memeriksa kesehatan sistem"""
    try:
        load_all_model_components()
        return jsonify({"status": "ok", "message": "Sistem berjalan normal"}), 200

    except ModelLoadingError as e:
        logging.error(f"Error loading model: {str(e)}")
        return jsonify(
            {
                "status": "error",
                "code": "MODEL_UNAVAILABLE",
                "solution": "Please run model training first",
            }
        ), 503

    except Exception as ex:
        logging.exception("Unexpected system error")
        return jsonify(
            {
                "status": "error",
                "code": "SERVER_ERROR",
                "detail": "Internal server problem",
            }
        ), 500


@pengujian_bp.route("/predict", methods=["POST"])
@limiter.limit("10 per minute")
def predict_sentimen():
    """Handle sentiment prediction requests"""
    try:
        # Validate input
        try:
            data = PredictionRequest(**request.get_json())
        except ValidationError as e:
            logger.warning(f"Invalid input: {str(e)}")
            return jsonify({"error": "Invalid input", "details": e.errors()}), 400

        # Load model components
        try:
            model, encoder, vectorizer = load_all_model_components()
        except ModelLoadingError as e:
            logger.error(f"Model loading failed: {str(e)}")
            return jsonify(
                {"error": str(e), "solution": "Contact system administrator"}
            ), 503

        # Process text
        try:
            ulasan = html.escape(data.ulasan.strip())
            terjemahan = translate_large_text(ulasan, "id")
            is_translated = terjemahan.strip() != ulasan.strip()

            hasil_prepro = proses_baris_aman(terjemahan)
            if not hasil_prepro or not hasil_prepro[-1]:
                logger.warning("Preprocessing returned empty result")
                return jsonify(
                    {
                        "error": "Text processing failed",
                        "detail": "Invalid preprocessing output",
                    }
                ), 400
        except Exception as e:
            logger.error(f"Text processing error: {str(e)}")
            return jsonify({"error": "Text processing failed", "detail": str(e)}), 500

        # Make prediction
        try:
            vector = vectorizer.transform([hasil_prepro[-1]])
            pred = model.predict(vector)
            label_prediksi = encoder.inverse_transform([np.argmax(pred)])[0]
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return jsonify({"error": "Prediction failed", "detail": str(e)}), 500

        # Verify label if provided
        label_match = None
        if data.label:
            valid_labels = {label.lower() for label in encoder.classes_}
            if data.label.lower() not in valid_labels:
                logger.warning(f"Invalid label provided: {data.label}")
                return jsonify(
                    {"error": "Invalid label", "valid_labels": list(valid_labels)}
                ), 400
            label_match = label_prediksi.lower() == data.label.lower()

        # Return successful response
        logger.info(f"Successful prediction: {label_prediksi}")
        return jsonify(
            {
                "result": {"prediction": label_prediksi, "label_match": label_match},
                "process": {
                    "translated": is_translated,
                    "preprocessed_text": hasil_prepro[-1],
                },
            }
        )

    except Exception as e:
        logger.exception("Unexpected prediction error")
        return jsonify(
            {"error": "System error", "detail": "Please try again later"}
        ), 500

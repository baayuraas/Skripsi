import os
import logging
import pickle
import numpy as np
from functools import lru_cache
from typing import Optional, Tuple, Any
from pydantic import BaseModel, ValidationError, constr, Field
from flask import Blueprint, request, jsonify, render_template
from keras.models import load_model
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from datetime import datetime
import scipy.sparse as sp
import nltk
from deep_translator import GoogleTranslator

# Download NLTK resources
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except Exception as e:
    print(f"NLTK resources download failed: {e}")

# IMPORT MODUL PREPROCESSING
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../preproses"))
from preprocessing_module import preprocessor

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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
TFIDF_PATH = os.path.join(BASE_DIR, "uploads", "tfidf", "tfidf_unlimited_model.pkl")

# Cache untuk waktu modifikasi file
file_modification_times = {}

# Mapping label dari frontend ke model
LABEL_MAPPING = {"positif": "baik", "negatif": "buruk"}


# Custom Exception
class ModelLoadingError(Exception):
    pass


class TranslationError(Exception):
    pass


class PreprocessingError(Exception):
    pass


class PredictionError(Exception):
    pass


# PERBAIKAN: Ubah batas maksimum karakter dari 1000 menjadi 2000
ConstrainedStr = constr(min_length=1, max_length=2000)


class PredictionRequest(BaseModel):
    # PERBAIKAN: Ubah batas maksimum karakter dari 1000 menjadi 2000
    ulasan: str = Field(..., min_length=1, max_length=2000)
    label: Optional[str] = None


# Translation implementation with deep_translator
def translate_large_text(text: str, target_lang: str) -> str:
    """
    Translate text to target language using Google Translate API via deep_translator.
    Falls back to original text if translation fails.
    """
    try:
        # Translate text
        translated_text = GoogleTranslator(source="auto", target=target_lang).translate(
            text
        )
        logger.info(
            f"Text translated successfully: {text[:50]}... -> {translated_text[:50]}..."
        )

        return translated_text

    except ImportError:
        logger.warning(
            "deep_translator package not installed. Please install with: pip install deep-translator"
        )
        # Fallback: return original text
        return text
    except Exception as e:
        logger.warning(f"Translation failed: {str(e)}. Using original text.")
        # Fallback: return original text
        return text


# GUNAKAN MODUL PREPROCESSING UNTUK FUNGSI PREPROCESSING
def proses_baris_aman(text: str) -> list:
    """
    Preprocess text menggunakan modul preprocessing yang SAMA dengan preproses
    Returns a list containing the preprocessed text.
    """
    try:
        # Gunakan modul preprocessing
        return preprocessor.proses_baris_aman(text)
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise PreprocessingError(f"Text preprocessing failed: {str(e)}")


@lru_cache(maxsize=1)
def load_all_model_components() -> Tuple[Any, Any, Any]:
    """
    Load model, label encoder, and TF-IDF vectorizer with caching.
    """
    # Check file existence
    if not os.path.exists(MODEL_PATH):
        raise ModelLoadingError(f"Model tidak ditemukan: {MODEL_PATH}")
    if not os.path.exists(LABEL_PATH):
        raise ModelLoadingError(f"LabelEncoder tidak ditemukan: {LABEL_PATH}")
    if not os.path.exists(TFIDF_PATH):
        raise ModelLoadingError(f"TF-IDF vectorizer tidak ditemukan: {TFIDF_PATH}")

    try:
        # Load components
        model = load_model(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")

        with open(LABEL_PATH, "rb") as f:
            encoder = pickle.load(f)
        logger.info(f"Label encoder loaded successfully from {LABEL_PATH}")

        with open(TFIDF_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        logger.info(f"TF-IDF vectorizer loaded successfully from {TFIDF_PATH}")

        # Verify that components are loaded correctly
        if model is None:
            raise ModelLoadingError("Model is None after loading")
        if encoder is None:
            raise ModelLoadingError("Encoder is None after loading")
        if vectorizer is None:
            raise ModelLoadingError("Vectorizer is None after loading")

        # Store modification times for cache validation
        global file_modification_times
        file_modification_times["model"] = os.path.getmtime(MODEL_PATH)
        file_modification_times["label"] = os.path.getmtime(LABEL_PATH)
        file_modification_times["tfidf"] = os.path.getmtime(TFIDF_PATH)

        return model, encoder, vectorizer

    except Exception as e:
        logger.error(f"Gagal memuat model: {str(e)}")
        raise ModelLoadingError(f"Gagal memuat komponen model: {str(e)}")


def check_model_files_updated() -> bool:
    """Check if model files have been updated since last load"""
    global file_modification_times

    if not file_modification_times:
        return True

    try:
        current_model_mtime = os.path.getmtime(MODEL_PATH)
        current_label_mtime = os.path.getmtime(LABEL_PATH)
        current_tfidf_mtime = os.path.getmtime(TFIDF_PATH)

        return (
            current_model_mtime != file_modification_times.get("model")
            or current_label_mtime != file_modification_times.get("label")
            or current_tfidf_mtime != file_modification_times.get("tfidf")
        )
    except Exception as e:
        logger.error(f"Error checking file modification times: {str(e)}")
        return True


def ensure_sparse_matrix_sorted(sparse_matrix):
    """
    Ensure that sparse matrix indices are sorted.
    This fixes the 'indices out of order' error.
    """
    if not hasattr(sparse_matrix, "sort_indices"):
        return sparse_matrix

    # Convert to CSR format if not already
    if not sp.isspmatrix_csr(sparse_matrix):
        sparse_matrix = sparse_matrix.tocsr()

    # Sort indices
    sparse_matrix.sort_indices()

    return sparse_matrix


def map_label_to_model(label: str) -> str:
    """Map label from frontend to model label"""
    if not label:
        return label

    # Convert to lowercase for case-insensitive comparison
    label_lower = label.lower()

    # Return mapped label if exists, otherwise return original
    return LABEL_MAPPING.get(label_lower, label_lower)


# ROUTES
@pengujian_bp.route("/", methods=["GET"])
def index():
    return render_template("data_pengujian.html", page_name="pengujian")


@pengujian_bp.route("/health", methods=["GET"])
def health_check():
    try:
        # Check if model files have been updated
        if check_model_files_updated():
            load_all_model_components.cache_clear()
            logger.info("Model cache cleared due to file updates")

        model, encoder, vectorizer = load_all_model_components()

        # Pastikan komponen sudah dimuat dengan benar
        if model is None or encoder is None or vectorizer is None:
            raise ModelLoadingError("Salah satu komponen model adalah None")

        # Test prediction dengan teks sederhana
        sample_text = "Testing system health"
        vector = vectorizer.transform([sample_text])

        # Ensure sparse matrix indices are sorted
        vector = ensure_sparse_matrix_sorted(vector)

        # Convert to dense array for model prediction
        vector_dense = vector.toarray()

        # Check vector shape
        if vector_dense.shape[1] != model.input_shape[1]:
            raise ModelLoadingError(
                f"Vector shape {vector_dense.shape} doesn't match model input shape {model.input_shape}"
            )

        # Make prediction (result is intentionally not used, just to test)
        model.predict(vector_dense)

        return jsonify(
            {
                "status": "ok",
                "message": "Sistem berjalan normal",
                "model_info": {
                    "model_path": MODEL_PATH,
                    "last_modified": datetime.fromtimestamp(
                        os.path.getmtime(MODEL_PATH)
                    ).isoformat(),
                    "input_shape": model.input_shape[1:]
                    if hasattr(model, "input_shape")
                    else "unknown",
                    "output_shape": model.output_shape[1:]
                    if hasattr(model, "output_shape")
                    else "unknown",
                },
                "label_classes": encoder.classes_.tolist()
                if hasattr(encoder, "classes_")
                else [],
                "label_mapping": LABEL_MAPPING,
            }
        ), 200

    except ModelLoadingError as e:
        logger.error(f"Error loading model: {e}")
        return jsonify(
            {
                "status": "error",
                "code": "MODEL_UNAVAILABLE",
                "solution": "Silakan jalankan proses training terlebih dahulu",
                "details": str(e),
            }
        ), 503

    except Exception as ex:
        logger.exception(f"Unexpected error: {ex}")
        return jsonify(
            {
                "status": "error",
                "code": "SERVER_ERROR",
                "detail": "Internal server error",
                "timestamp": datetime.now().isoformat(),
            }
        ), 500


@pengujian_bp.route("/predict", methods=["POST"])
@limiter.limit("10 per minute")
def predict_sentimen():
    # Check content type
    if not request.is_json:
        return jsonify(
            {
                "status": "error",
                "code": "INVALID_CONTENT_TYPE",
                "message": "Content-Type must be application/json",
            }
        ), 400

    try:
        # Validate input
        try:
            data = PredictionRequest(**request.get_json())
        except ValidationError as e:
            # PERBAIKAN: Menambahkan penanganan error yang lebih spesifik dan aman
            error_details = []
            for error in e.errors():
                if error["type"] == "string_too_long":
                    if "ctx" in error and "limit_value" in error["ctx"]:
                        error_details.append(
                            f"Field {error['loc'][0]} melebihi batas maksimum {error['ctx']['limit_value']} karakter"
                        )
                    else:
                        error_details.append(
                            f"Field {error['loc'][0]} melebihi batas maksimum karakter"
                        )
                else:
                    error_details.append(error["msg"])

            return jsonify(
                {
                    "status": "error",
                    "code": "VALIDATION_ERROR",
                    "message": "Input tidak valid",
                    "details": error_details,
                }
            ), 400

        # Load model components
        try:
            model, encoder, vectorizer = load_all_model_components()

            # Pastikan komponen sudah dimuat dengan benar
            if model is None or encoder is None or vectorizer is None:
                raise ModelLoadingError("Salah satu komponen model adalah None")

        except ModelLoadingError as e:
            return jsonify(
                {"status": "error", "code": "MODEL_UNAVAILABLE", "message": str(e)}
            ), 503

        # PERBAIKAN: Hapus html.escape() dari pemrosesan input
        # Sanitize and validate input
        ulasan = data.ulasan.strip()  # DIUBAH: menghapus html.escape()
        if not ulasan:
            return jsonify(
                {
                    "status": "error",
                    "code": "EMPTY_INPUT",
                    "message": "Ulasan tidak boleh kosong setelah pembersihan",
                }
            ), 400

        # Translation
        is_translated = False
        terjemahan = ulasan
        try:
            # Coba terjemahkan ke Bahasa Indonesia
            terjemahan = translate_large_text(ulasan, "id")
            is_translated = terjemahan.strip() != ulasan.strip()

            # Log hasil terjemahan
            if is_translated:
                logger.info(
                    f"Text translated: {ulasan[:100]}... -> {terjemahan[:100]}..."
                )
            else:
                logger.info("No translation needed or translation returned same text")

        except Exception as e:
            logger.warning(f"Translation skipped: {str(e)}")
            terjemahan = ulasan
            is_translated = False

        # Preprocessing menggunakan MODUL PREPROCESSING yang SAMA dengan preproses
        try:
            hasil_prepro = proses_baris_aman(terjemahan)
            if not hasil_prepro or not hasil_prepro[0]:
                return jsonify(
                    {
                        "status": "error",
                        "code": "PREPROCESSING_ERROR",
                        "message": "Preprocessing gagal menghasilkan teks valid",
                    }
                ), 400
        except PreprocessingError as e:
            return jsonify(
                {"status": "error", "code": "PREPROCESSING_ERROR", "message": str(e)}
            ), 400

        # Vectorization and prediction
        try:
            # Transform text to vector
            vector = vectorizer.transform([hasil_prepro[0]])

            # Ensure sparse matrix indices are sorted
            vector = ensure_sparse_matrix_sorted(vector)

            # Convert to dense array for model prediction
            vector_dense = vector.toarray()

            # Check if vector is empty
            if vector_dense.shape[1] == 0:
                raise PredictionError("Vector is empty after transformation")

            # Check if vector shape matches model input
            if (
                hasattr(model, "input_shape")
                and vector_dense.shape[1] != model.input_shape[1]
            ):
                raise PredictionError(
                    f"Vector shape {vector_dense.shape} doesn't match model input shape {model.input_shape}"
                )

            # Make prediction
            pred = model.predict(vector_dense)

            # Check if prediction is valid
            if pred is None or len(pred) == 0:
                raise PredictionError("Prediction returned empty result")

            label_prediksi = encoder.inverse_transform([np.argmax(pred)])[0]
            confidence = float(np.max(pred))

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            return jsonify(
                {
                    "status": "error",
                    "code": "PREDICTION_ERROR",
                    "message": "Terjadi kesalahan saat melakukan prediksi",
                    "details": str(e),
                }
            ), 500

        # Validate provided label if any
        label_match = None
        if data.label:
            # Map label from frontend to model label
            mapped_label = map_label_to_model(data.label)

            valid_labels = {label.lower() for label in encoder.classes_}
            if mapped_label not in valid_labels:
                return jsonify(
                    {
                        "status": "error",
                        "code": "INVALID_LABEL",
                        "message": "Label tidak valid",
                        "valid_labels": list(valid_labels),
                        "provided_label": data.label,
                        "mapped_label": mapped_label,
                    }
                ), 400
            label_match = label_prediksi.lower() == mapped_label

        # Return successful response
        return jsonify(
            {
                "status": "success",
                "result": {
                    "prediction": label_prediksi,
                    "confidence": confidence,
                    "label_match": label_match,
                },
                "process": {
                    "translated": is_translated,
                    "original_text": ulasan,
                    "translation_text": terjemahan if is_translated else ulasan,
                    "preprocessed_text": hasil_prepro[0],
                },
                "model_info": {
                    "model_type": "MLP",
                    "input_length": vector_dense.shape[1],
                },
            }
        )

    except Exception as e:
        logger.exception(f"Unexpected error in prediction: {e}")
        return jsonify(
            {
                "status": "error",
                "code": "INTERNAL_ERROR",
                "message": "Terjadi kesalahan internal pada sistem",
                "timestamp": datetime.now().isoformat(),
                "details": str(e),
            }
        ), 500


# Endpoint untuk debugging preprocessing
@pengujian_bp.route("/debug_preprocessing", methods=["POST"])
def debug_preprocessing():
    """Endpoint untuk debugging preprocessing pada teks tertentu"""
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Tidak ada teks yang dikirim."}), 400

        teks = data["text"]
        logger.info(f"Debug Preprocessing: Memproses teks: {teks}")

        # Lakukan preprocessing dengan debug mode menggunakan MODUL
        hasil = preprocessor.proses_preprocessing_standar(teks, debug=True)

        return jsonify(
            {
                "status": "success",
                "input_text": teks,
                "processing_steps": {
                    "clean": hasil[0],
                    "folded": hasil[1],
                    "tokens": hasil[2],
                    "stopwords_removed": hasil[3],
                    "normalized": hasil[4],
                    "stemmed": hasil[5],
                    "final": hasil[6],
                },
                "dictionary_info": {
                    "normalization_dict_size": len(preprocessor.normalisasi_dict),
                    "game_terms_size": len(preprocessor.game_terms),
                    "whitelist_size": len(preprocessor.kata_id_pasti),
                },
            }
        )
    except Exception as e:
        logger.error(f"Error debug_preprocessing: {e}")
        return jsonify({"error": str(e)}), 500


# Endpoint untuk mengecek kesehatan preprocessing module
@pengujian_bp.route("/preprocessing_health", methods=["GET"])
def preprocessing_health():
    """Endpoint untuk mengecek kesehatan modul preprocessing"""
    try:
        # Test dengan contoh teks
        test_text = "gue mau makan bgtt ntar klo bs"
        hasil = preprocessor.proses_preprocessing_standar(test_text, debug=False)

        return jsonify(
            {
                "status": "success",
                "message": "Modul preprocessing berjalan normal",
                "test_preprocessing": {
                    "input": test_text,
                    "output": hasil[6],
                    "steps": {
                        "clean": hasil[0],
                        "folded": hasil[1],
                        "tokens": len(hasil[2]),
                        "stopwords_removed": len(hasil[3]),
                        "normalized": len(hasil[4]),
                        "stemmed": len(hasil[5]),
                    },
                },
                "module_info": {
                    "initialized": preprocessor._initialized,
                    "dictionaries_loaded": {
                        "normalization": len(preprocessor.normalisasi_dict) > 0,
                        "stemming": len(preprocessor.stemming_dict) > 0,
                        "game_terms": len(preprocessor.game_terms) > 0,
                        "stopwords_id": len(preprocessor.stop_words_id) > 0,
                        "stopwords_ing": len(preprocessor.stop_words_ing) > 0,
                    },
                },
            }
        )
    except Exception as e:
        logger.error(f"Error preprocessing_health: {e}")
        return jsonify({"error": str(e)}), 500

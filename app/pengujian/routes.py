import os
import logging
import pickle
import re
import numpy as np
from functools import lru_cache
from typing import Optional, Tuple, Any
from pydantic import BaseModel, ValidationError, constr, Field
from flask import Blueprint, request, jsonify, render_template
from keras.models import load_model
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
from deep_translator import GoogleTranslator

# Download NLTK resources
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except Exception as e:
    print(f"NLTK resources download failed: {e}")

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
TFIDF_PATH = os.path.join(BASE_DIR, "uploads", "tfidf", "tfidf_train.pkl")

# Perbaikan: Path ke direktori preproses yang berisi file TXT
TXT_DIR = os.path.join(BASE_DIR, "app", "preproses")

# Daftar file txt yang diperlukan
required_txt_files = [
    "stopword_list.txt",
    "stopword_list_ing.txt",
    "normalisasi_list.txt",
    "stemming_list.txt",
    "game_term.txt",
    "kata_tidak_relevan.txt",
    "kata_ambigu.txt",
]

# Validasi file yang diperlukan
for file_name in required_txt_files:
    file_path = os.path.join(TXT_DIR, file_name)
    if not os.path.exists(file_path):
        logger.warning(f"File {file_name} tidak ditemukan di {TXT_DIR}")
    else:
        logger.info(f"File {file_name} ditemukan")

# Cache untuk waktu modifikasi file
file_modification_times = {}

# Mapping label dari frontend ke model
LABEL_MAPPING = {"positif": "baik", "negatif": "buruk"}


# Fungsi untuk memuat stopword dengan format yang konsisten
def load_stopwords(filename):
    """Memuat stopword dari file dan memastikan format konsisten"""
    stopwords_set = set()
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                # Hapus karakter tidak diinginkan dan pastikan hanya huruf
                word = re.sub(r"[^a-z]", "", word)
                if (
                    word and len(word) > 1
                ):  # Hanya tambahkan jika tidak kosong dan panjang > 1
                    stopwords_set.add(word)
        logger.info(f"Loaded {len(stopwords_set)} kata dari {filename}")
    except Exception as e:
        logger.error(f"Error load_stopwords {filename}: {e}")
    return stopwords_set


# Inisialisasi stopword factory
stopword_factory = StopWordRemoverFactory()
stemmer = StemmerFactory().create_stemmer()

# Load stopword dasar Indonesia
stop_words_id = set(stopword_factory.get_stop_words()).union(
    set(stopwords.words("indonesian"))
)

# Load stopword Indonesia dari file
id_stopwords_file = os.path.join(TXT_DIR, "stopword_list.txt")
if os.path.exists(id_stopwords_file):
    id_stopwords = load_stopwords(id_stopwords_file)
    stop_words_id.update(id_stopwords)

# Load stopword Inggris dari file
stopword_ing_file = os.path.join(TXT_DIR, "stopword_list_ing.txt")
stop_words_ing = set()
if os.path.exists(stopword_ing_file):
    stop_words_ing = load_stopwords(stopword_ing_file)
else:
    # Fallback ke stopword Inggris dari NLTK jika file tidak ada
    stop_words_ing = set(stopwords.words("english"))
    logger.warning("Using NLTK English stopwords as fallback")

logger.info(f"Total stopwords Indonesia: {len(stop_words_id)}")
logger.info(f"Total stopwords Inggris: {len(stop_words_ing)}")

# Load normalisasi dictionary
normalisasi_dict = {}
normalisasi_file = os.path.join(TXT_DIR, "normalisasi_list.txt")
if os.path.exists(normalisasi_file):
    with open(normalisasi_file, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                # Pastikan format konsisten (huruf kecil)
                k = k.strip().lower()
                v = v.strip().lower()
                normalisasi_dict[k] = v
    logger.info(f"Loaded {len(normalisasi_dict)} entries dari normalisasi_list.txt")

# Load stemming_list.txt
stemming_dict = {}
stemming_file = os.path.join(TXT_DIR, "stemming_list.txt")
if os.path.exists(stemming_file):
    with open(stemming_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                parts = line.split("=", 1)
                k = parts[0].strip().lower()
                v = parts[1].strip().lower()
                stemming_dict[k] = v

                # Tambahkan juga varian dengan normalisasi pengulangan huruf
                k_normalized = re.sub(r"(.)\1+", r"\1", k)
                if k_normalized != k and k_normalized not in stemming_dict:
                    stemming_dict[k_normalized] = v

    logger.info(f"Loaded {len(stemming_dict)} pasangan dari stemming_list.txt")
else:
    logger.warning("File stemming_list.txt tidak ditemukan, gunakan default Sastrawi.")

# Load game terms
game_terms = set()
game_terms_file = os.path.join(TXT_DIR, "game_term.txt")
if os.path.exists(game_terms_file):
    with open(game_terms_file, "r", encoding="utf-8") as f:
        for line in f:
            term = line.strip().lower()
            if term:
                game_terms.add(term)
    logger.info(f"Loaded {len(game_terms)} game terms")

# Load kata tidak relevan
kata_tidak_relevan = set()
kata_tidak_relevan_file = os.path.join(TXT_DIR, "kata_tidak_relevan.txt")
if os.path.exists(kata_tidak_relevan_file):
    with open(kata_tidak_relevan_file, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                kata_tidak_relevan.add(word)
    logger.info(f"Loaded {len(kata_tidak_relevan)} kata tidak relevan")

# Load kata ambigu (whitelist)
kata_id_pasti = set()
kata_ambigu_file = os.path.join(TXT_DIR, "kata_ambigu.txt")
if os.path.exists(kata_ambigu_file):
    try:
        with open(kata_ambigu_file, "r", encoding="utf-8") as f:
            kata_id_pasti = {line.strip().lower() for line in f if line.strip()}
        logger.info(f"Loaded {len(kata_id_pasti)} kata dari kata_ambigu.txt")
    except Exception as e:
        logger.error(f"Error baca whitelist: {e}")
else:
    logger.warning("File kata_ambigu.txt tidak ditemukan, whitelist kosong.")

# Pre-compile regex patterns - DIPERBAIKI
emoji_pattern = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags (iOS)
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "]+",
    flags=re.UNICODE,
)

# Pattern yang diperbaiki
special_char_pattern = re.compile(r"[-–—…\"»«]")  # Diperbaiki
bracket_pattern = re.compile(r"\[.*?\]")
url_pattern = re.compile(r"http\S+")  # Diperbaiki
digit_pattern = re.compile(r"\b\d+\b")
non_word_pattern = re.compile(r"[^\w\s@#]")
whitespace_pattern = re.compile(r"\s+")
repeated_word_pattern = re.compile(r"\b(\w{3,}?)(?:\1)\b")  # Diperbaiki
word_pattern = re.compile(r"\b\w+\b")

# Pattern untuk membersihkan token tidak diinginkan
number_unit_pattern = re.compile(r"^\d+[a-z]*\d*[a-z]*$")
short_word_pattern = re.compile(r"^\w{1,2}$")
mixed_alnum_pattern = re.compile(r"^[a-z]+\d+|\d+[a-z]+$")


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


# Fungsi untuk membersihkan teks
def bersihkan_terjemahan(teks: str) -> str:
    if pd.isna(teks) or not isinstance(teks, str):
        return ""

    teks_asli = teks

    try:
        teks = emoji_pattern.sub(" ", teks)
        teks = special_char_pattern.sub(" ", teks)
        teks = bracket_pattern.sub(" ", teks)
        teks = url_pattern.sub(" ", teks)
        teks = digit_pattern.sub(" ", teks)
        teks = non_word_pattern.sub(" ", teks)
        teks = whitespace_pattern.sub(" ", teks).strip()

        if not teks.strip():
            teks = emoji_pattern.sub(" ", teks_asli)
            teks = url_pattern.sub(" ", teks)
            teks = whitespace_pattern.sub(" ", teks).strip()

        return teks
    except Exception as e:
        logger.error(f"Error in bersihkan_terjemahan: {e}")
        return whitespace_pattern.sub(" ", teks_asli).strip()


# Fungsi untuk membersihkan token tidak diinginkan
def bersihkan_token(tokens):
    """Membersihkan token-token yang tidak diinginkan seperti angka dengan satuan, kata sangat pendek, dll."""
    hasil = []
    for token in tokens:
        # Normalisasi pengulangan huruf terlebih dahulu
        normalized_token = re.sub(r"(.)\1+", r"\1", token)

        # Skip token yang sesuai dengan pola tidak diinginkan
        if (
            number_unit_pattern.match(normalized_token)
            or short_word_pattern.match(normalized_token)
            or mixed_alnum_pattern.match(normalized_token)
        ):
            continue

        # Skip token yang hanya terdiri dari angka
        if normalized_token.isdigit():
            continue

        # Gunakan token yang sudah dinormalisasi
        hasil.append(normalized_token)

    return hasil


# Fungsi normalisasi teks
def normalisasi_teks(words):
    hasil = []
    for w in words:
        wl = w.lower()
        if wl in normalisasi_dict:
            mapped = normalisasi_dict[wl]
            hasil.extend(mapped.split())
        else:
            hasil.append(wl)
    return hasil


# Fungsi Stopword Removal dengan Prioritas yang Diperbarui
def hapus_stopword(words, debug=False):
    # Bersihkan token tidak diinginkan terlebih dahulu
    words = bersihkan_token(words)

    if debug:
        logger.debug(f"Kata sebelum filter: {words}")

    result = []
    for w in words:
        # Pengecualian: Jika kata adalah game term, selalu pertahankan
        if w in game_terms:
            if debug:
                logger.debug(f"Menyimpan game term: '{w}'")
            result.append(w)
            continue

        # Prioritas 1: Cek kata tidak relevan terlebih dahulu (jika bukan stopword Indonesia)
        if w not in stop_words_id and w in kata_tidak_relevan:
            if debug:
                logger.debug(f"Menghapus kata tidak relevan: '{w}'")
            continue

        # Prioritas 2: Cek stopword Inggris
        if w in stop_words_ing:
            if debug:
                logger.debug(f"Menghapus stopword Inggris: '{w}'")
            continue

        # Prioritas 3: Cek stopword Indonesia
        if w in stop_words_id:
            if debug:
                logger.debug(f"Menghapus stopword Indonesia: '{w}'")
            continue

        # Jika bukan stopword atau kata tidak relevan, simpan
        if debug:
            logger.debug(f"Menyimpan kata: '{w}'")
        result.append(w)

    if debug:
        logger.debug(f"Kata setelah filter: {result}")

    return result


# Fungsi Stemming yang Diperbarui
def stemming_teks(words, debug=False):
    hasil = []

    for w in words:
        wl = w.lower()
        wl_normalized = re.sub(r"(.)\1+", r"\1", wl)

        # 1. PRIORITAS PERTAMA: Cek kamus custom (stemming_list.txt)
        if wl in stemming_dict:
            mapped = stemming_dict[wl]
            hasil.extend(mapped.split())
        elif wl_normalized in stemming_dict:
            mapped = stemming_dict[wl_normalized]
            hasil.extend(mapped.split())
        else:
            # 2. PRIORITAS KEDUA: Gunakan Sastrawi
            stemmed_sastrawi = stemmer.stem(wl_normalized)

            if stemmed_sastrawi != wl_normalized:
                hasil.append(stemmed_sastrawi)
            else:
                # Jika semua metode gagal, gunakan kata asli yang sudah dinormalisasi
                hasil.append(wl_normalized)

    return hasil


# Improved preprocessing function menggunakan program preprocessing yang lengkap
def proses_baris_aman(text: str) -> list:
    """
    Preprocess text menggunakan program preprocessing yang lengkap.
    Returns a list containing the preprocessed text.
    """
    try:
        # Bersihkan teks
        clean = bersihkan_terjemahan(text)
        folded = clean.lower()

        # Tokenisasi
        try:
            token = word_tokenize(folded)
        except Exception:
            token = word_pattern.findall(folded)

        # Bersihkan token tidak diinginkan
        token = bersihkan_token(token)

        # Normalisasi teks
        norm = normalisasi_teks(token) if token else []

        # Hapus stopword
        stop = hapus_stopword(norm) if norm else []

        # Stemming
        stem = stemming_teks(stop) if stop else []

        # Gabungkan hasil
        hasil = " ".join(stem) if stem else folded

        return [hasil]
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

        # Preprocessing menggunakan program preprocessing yang lengkap
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

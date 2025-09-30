import json
import os
import re
import pandas as pd
from flask import Blueprint, request, jsonify, render_template, send_file
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from markupsafe import escape
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from deep_translator import GoogleTranslator
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from functools import lru_cache
import random
from collections import Counter
import logging

# --- Konfigurasi dasar ---
prepro_bp = Blueprint(
    "prepro",
    __name__,
    url_prefix="/prepro",
    template_folder="templates",
    static_folder="static",
)

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Pastikan hasil deteksi bahasa konsisten
DetectorFactory.seed = 0

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "preproses")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
PREPRO_CSV_PATH = os.path.join(UPLOAD_FOLDER, "processed_data.csv")
TXT_DIR = os.path.dirname(os.path.abspath(__file__))

# Deteksi environment Flask - untuk menghindari inisialisasi ganda
IS_MAIN_PROCESS = os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not os.environ.get(
    "WERKZEUG_RUN_MAIN"
)

# Konfigurasi retry dan jeda
MAX_RETRIES = 3
INITIAL_DELAY = 1
MAX_DELAY = 10
REQUEST_DELAY = 0.5

# Daftar file txt yang diperlukan
required_txt_files = [
    "stopword_list.txt",
    "stopword_list_ing.txt",
    "normalisasi_list.txt",
    "stemming_list.txt",
    "game_term.txt",
    "kata_tidak_relevan.txt",
    "kata_ambigu.txt",
    "normalisasi_stopword_list.txt",
]

# Flag untuk menandai apakah inisialisasi sudah dilakukan
_INITIALIZED = False

# Variabel global yang akan diinisialisasi
stop_words_id = set()
stop_words_ing = set()
normalisasi_dict = {}
stemming_dict = {}
game_terms = set()
kata_tidak_relevan = set()
kata_id_pasti = set()
normalisasi_stopword_set = set()


# --- Fungsi inisialisasi sekali pakai ---
def initialize_preprocessing_data():
    """Fungsi untuk inisialisasi data preprocessing sekali saja"""
    global _INITIALIZED, stop_words_id, stop_words_ing, normalisasi_dict, stemming_dict
    global game_terms, kata_tidak_relevan, kata_id_pasti, normalisasi_stopword_set

    if _INITIALIZED:
        logging.info("✅ Data preprocessing sudah diinisialisasi sebelumnya")
        return

    if not IS_MAIN_PROCESS:
        logging.info("⏸️  Skip inisialisasi: Ini adalah proses reloader Flask")
        return

    logging.info("🔄 Memulai inisialisasi data preprocessing...")

    # Validasi file yang diperlukan
    for file_name in required_txt_files:
        file_path = os.path.join(TXT_DIR, file_name)
        if not os.path.exists(file_path):
            logging.warning(f"File {file_name} tidak ditemukan di {TXT_DIR}")
        else:
            logging.info(f"File {file_name} ditemukan")

    # Inisialisasi stopword factory
    stopword_factory = StopWordRemoverFactory()

    # Load stopword dasar Indonesia
    stop_words_id = set(stopword_factory.get_stop_words()).union(
        set(stopwords.words("indonesian"))
    )

    # Load stopword Indonesia dari file dengan validasi
    id_stopwords = load_stopwords(os.path.join(TXT_DIR, "stopword_list.txt"))
    stop_words_id.update(id_stopwords)

    # Validasi stopword Indonesia
    expected_id_stopwords = {
        "yang",
        "dan",
        "di",
        "dari",
        "ke",
        "pada",
        "ini",
        "itu",
        "dengan",
        "untuk",
    }
    missing_id = expected_id_stopwords - stop_words_id
    if missing_id:
        logging.warning(f"Stopword ID yang hilang: {missing_id}")
        stop_words_id.update(missing_id)

    # Load stopword Inggris dari file
    stopword_ing_file = os.path.join(TXT_DIR, "stopword_list_ing.txt")
    if os.path.exists(stopword_ing_file):
        stop_words_ing = load_stopwords(stopword_ing_file)
    else:
        stop_words_ing = set(stopwords.words("english"))
        logging.info("Using NLTK English stopwords as fallback")

    # Validasi stopword Inggris
    expected_ing_stopwords = {
        "the",
        "and",
        "is",
        "in",
        "to",
        "of",
        "a",
        "for",
        "on",
        "with",
    }
    missing_ing = expected_ing_stopwords - stop_words_ing
    if missing_ing:
        logging.warning(f"Stopword ING yang hilang: {missing_ing}")
        stop_words_ing.update(missing_ing)

    logging.info(f"Total stopwords Indonesia: {len(stop_words_id)}")
    logging.info(f"Total stopwords Inggris: {len(stop_words_ing)}")

    # Load normalisasi dictionary
    normalisasi_dict.clear()
    normalisasi_file = os.path.join(TXT_DIR, "normalisasi_list.txt")
    if os.path.exists(normalisasi_file):
        with open(normalisasi_file, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    k = k.strip().lower()
                    v = v.strip().lower()
                    normalisasi_dict[k] = v
        logging.info(
            f"Loaded {len(normalisasi_dict)} entries dari normalisasi_list.txt"
        )
    else:
        logging.warning("File normalisasi_list.txt tidak ditemukan")

    # Load stemming_list.txt
    stemming_dict.clear()
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

                    k_normalized = re.sub(r"(.)\1+", r"\1", k)
                    if k_normalized != k and k_normalized not in stemming_dict:
                        stemming_dict[k_normalized] = v
        logging.info(f"Loaded {len(stemming_dict)} pasangan dari stemming_list.txt")
    else:
        logging.warning(
            "File stemming_list.txt tidak ditemukan, gunakan default Sastrawi."
        )

    # Load game terms
    game_terms.clear()
    game_terms_file = os.path.join(TXT_DIR, "game_term.txt")
    if os.path.exists(game_terms_file):
        with open(game_terms_file, "r", encoding="utf-8") as f:
            for line in f:
                term = line.strip().lower()
                if term:
                    game_terms.add(term)
        logging.info(f"Loaded {len(game_terms)} game terms")
    else:
        logging.warning("File game_term.txt tidak ditemukan")

    # Load kata tidak relevan
    kata_tidak_relevan.clear()
    tidak_relevan_file = os.path.join(TXT_DIR, "kata_tidak_relevan.txt")
    if os.path.exists(tidak_relevan_file):
        with open(tidak_relevan_file, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    kata_tidak_relevan.add(word)
        logging.info(f"Loaded {len(kata_tidak_relevan)} kata tidak relevan")
    else:
        logging.warning("File kata_tidak_relevan.txt tidak ditemukan")

    # Load kata ambigu (whitelist)
    kata_id_pasti.clear()
    whitelist_file = os.path.join(TXT_DIR, "kata_ambigu.txt")
    if os.path.exists(whitelist_file):
        try:
            with open(whitelist_file, "r", encoding="utf-8") as f:
                kata_id_pasti = {line.strip().lower() for line in f if line.strip()}
            logging.info(f"Loaded {len(kata_id_pasti)} kata dari kata_ambigu.txt")
        except Exception as e:
            logging.error(f"Error baca whitelist: {e}")
    else:
        logging.warning("File kata_ambigu.txt tidak ditemukan, whitelist kosong.")

    # Load kata dari normalisasi_stopword_list.txt
    normalisasi_stopword_set.clear()
    normalisasi_stopword_file = os.path.join(TXT_DIR, "normalisasi_stopword_list.txt")
    if os.path.exists(normalisasi_stopword_file):
        try:
            with open(normalisasi_stopword_file, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        normalisasi_stopword_set.add(word)
            logging.info(
                f"Loaded {len(normalisasi_stopword_set)} kata dari normalisasi_stopword_list.txt"
            )
        except Exception as e:
            logging.error(f"Error baca normalisasi_stopword_list.txt: {e}")
    else:
        logging.warning("File normalisasi_stopword_list.txt tidak ditemukan")

    _INITIALIZED = True
    logging.info("✅ Inisialisasi data preprocessing selesai")


def load_stopwords(filename):
    """Memuat stopword dari file dan memastikan format konsisten"""
    stopwords_set = set()
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                word = re.sub(r"[^a-z]", "", word)
                if word and len(word) > 1:
                    stopwords_set.add(word)
        logging.info(f"Loaded {len(stopwords_set)} kata dari {filename}")
    except Exception as e:
        logging.error(f"Error load_stopwords {filename}: {e}")
    return stopwords_set


# Panggil fungsi inisialisasi
initialize_preprocessing_data()

# Inisialisasi stemmer Sastrawi
stemmer = StemmerFactory().create_stemmer()

CHUNK_SIZE = 1000
MAX_FILE_SIZE = 2 * 1024 * 1024
CACHE_FOLDER = os.path.join(BASE_DIR, "uploads", "preproses")
CACHE_FILE = os.path.join(CACHE_FOLDER, "cache_translate.json")
LANGUAGE_CACHE_FILE = os.path.join(CACHE_FOLDER, "cache_language.json")

# --- Cache untuk deteksi bahasa ---
language_cache = {}
if os.path.exists(LANGUAGE_CACHE_FILE):
    try:
        with open(LANGUAGE_CACHE_FILE, "r", encoding="utf-8") as f:
            language_cache = json.load(f)
    except Exception:
        language_cache = {}


def simpan_cache_bahasa():
    try:
        os.makedirs(os.path.dirname(LANGUAGE_CACHE_FILE), exist_ok=True)
        with open(LANGUAGE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(language_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Error simpan cache bahasa: {e}")


def is_error_message(text):
    """Mendeteksi apakah teks mengandung pesan error"""
    if not text or not isinstance(text, str):
        return False

    if len(text.strip()) < 10:
        return False

    error_patterns = [
        r"error\s*\d+",
        r"server\s*error",
        r"try\s*later",
        r"internal\s*error",
        r"bad\s*request",
        r"gateway",
        r"timeout",
        r"service\s*unavailable",
        r"connection\s*refused",
        r"too\s*many\s*requests",
        r"quota\s*exceeded",
        r"api\s*key",
        r"limit\s*exceeded",
        r"connection\s*lost",
        r"network\s*error",
        r"socket",
        r"connection\s*reset",
        r"failed\s*to\s*connect",
    ]

    text_lower = text.lower()
    for pattern in error_patterns:
        if re.search(pattern, text_lower):
            return True
    return False


def simpan_cache_ke_file():
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(terjemahan_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Error simpan cache: {e}")


# --- Cache translate ---
terjemahan_cache = {}
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            terjemahan_cache = json.load(f)

        keys_to_remove = []
        for key, value in terjemahan_cache.items():
            if is_error_message(value) or is_error_message(key):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del terjemahan_cache[key]

        if keys_to_remove:
            logging.info(f"Menghapus {len(keys_to_remove)} entri error dari cache")
            simpan_cache_ke_file()
    except Exception:
        terjemahan_cache = {}

# --- Optimasi: Pre-compile regex patterns ---
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

special_char_pattern = re.compile(r"[-–—…\"»«]")
bracket_pattern = re.compile(r"\[.*?\]")
url_pattern = re.compile(r"http\S+")
digit_pattern = re.compile(r"\b\d+\b")
non_word_pattern = re.compile(r"[^\w\s@#]")
whitespace_pattern = re.compile(r"\s+")
repeated_word_pattern = re.compile(r"\b(\w{3,}?)(?:\1)\b")
sentence_split_pattern = re.compile(r"(?<=[.!?]) +|\n")
word_pattern = re.compile(r"\b\w+\b")

# --- Pattern untuk membersihkan token tidak diinginkan ---
number_unit_pattern = re.compile(r"^\d+[a-z]*\d*[a-z]*$")
short_word_pattern = re.compile(r"^\w{1,2}$")
mixed_alnum_pattern = re.compile(r"^[a-z]+\d+|\d+[a-z]+$")


# --- Optimasi: Gunakan LRU cache untuk fungsi yang sering dipanggil ---
@lru_cache(maxsize=10000)
def cached_stemmer_stem(word):
    return stemmer.stem(word)


@lru_cache(maxsize=10000)
def cached_detect_language(text):
    """Cache untuk deteksi bahasa dengan handling exception"""
    if not text or len(text.strip()) < 3:
        return "id"

    text_lower = text.lower()
    if text_lower in language_cache:
        return language_cache[text_lower]

    try:
        lang = detect(text)
        language_cache[text_lower] = lang
        return lang
    except LangDetectException:
        language_cache[text_lower] = "id"
        return "id"


def normalize_repeated_letters(word):
    """Mengurangi pengulangan huruf yang berlebihan menjadi maksimal 1 huruf"""
    if len(word) <= 2:
        return word
    normalized = re.sub(r"(.)\1+", r"\1", word)
    return normalized


def hapus_kata_normalisasi_stopword(words, debug=False):
    """Menghapus kata yang sama seperti yang ada dalam normalisasi_stopword_list.txt"""
    if not words:
        return []

    if debug:
        logging.debug(f"Kata sebelum hapus normalisasi_stopword: {words}")

    # Filter kata yang tidak ada dalam normalisasi_stopword_set
    result = [word for word in words if word not in normalisasi_stopword_set]

    if debug:
        removed = set(words) - set(result)
        if removed:
            logging.debug(f"Menghapus kata dari normalisasi_stopword_list: {removed}")
        logging.debug(f"Kata setelah hapus normalisasi_stopword: {result}")

    return result


def translate_with_retry(text, source="auto", target="id", max_len=5000):
    """Fungsi terjemahan dengan mekanisme retry untuk menangani connection lost"""
    if is_error_message(text):
        return text.lower()

    time.sleep(REQUEST_DELAY + random.uniform(0, 0.5))

    delay = INITIAL_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            translator = GoogleTranslator(source=source, target=target)
            hasil = []

            # Pecah teks berdasarkan tanda baca atau newline
            kalimat = sentence_split_pattern.split(text)

            buffer = ""
            for k in kalimat:
                if len(buffer) + len(k) < max_len:
                    buffer += " " + k
                else:
                    if buffer.strip():
                        try:
                            translated = translator.translate(buffer.strip())
                            if not is_error_message(translated):
                                hasil.append(translated)
                            else:
                                hasil.append(buffer.strip())
                        except Exception as e:
                            logging.error(f"Error translate chunk: {e}")
                            hasil.append(buffer.strip())
                    buffer = k

            if buffer.strip():
                try:
                    translated = translator.translate(buffer.strip())
                    if not is_error_message(translated):
                        hasil.append(translated)
                    else:
                        hasil.append(buffer.strip())
                except Exception as e:
                    logging.error(f"Error translate chunk: {e}")
                    hasil.append(buffer.strip())

            return " ".join(hasil)

        except Exception as e:
            error_msg = str(e).lower()
            logging.warning(
                f"Attempt {attempt + 1}/{MAX_RETRIES} Error dalam terjemahan: {e}"
            )

            is_connection_error = any(
                term in error_msg
                for term in [
                    "connection",
                    "network",
                    "socket",
                    "reset",
                    "timeout",
                    "refused",
                ]
            )

            if not is_connection_error or attempt == MAX_RETRIES - 1:
                raise e

            logging.info(f"Menunggu {delay} detik sebelum mencoba lagi...")
            time.sleep(delay)
            delay = min(MAX_DELAY, delay * 2 + random.uniform(0, 1))

    return text.lower()


# --- Utility ---
def register_template_filters(app):
    app.jinja_env.filters["sanitize"] = escape


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
        logging.error(f"Error bersihkan_terjemahan: {e}")
        return whitespace_pattern.sub(" ", teks_asli).strip()


def hapus_kata_ulang(word):
    return repeated_word_pattern.sub(r"\1", word)


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


def bersihkan_token(tokens):
    """Membersihkan token-token yang tidak diinginkan"""
    hasil = []
    for token in tokens:
        normalized_token = normalize_repeated_letters(token)

        if (
            number_unit_pattern.match(normalized_token)
            or short_word_pattern.match(normalized_token)
            or mixed_alnum_pattern.match(normalized_token)
        ):
            continue

        if normalized_token.isdigit():
            continue

        hasil.append(normalized_token)
    return hasil


def hapus_stopword(words, debug=False):
    """Fungsi stopword removal yang diperbaiki"""
    if not words:
        return []

    words = bersihkan_token(words)

    if debug:
        logging.debug(f"Kata sebelum filter stopword: {words}")

    result = []
    stopword_removed = []

    for w in words:
        w_lower = w.lower()

        # Pengecualian 1: Game terms
        if w_lower in game_terms:
            if debug:
                logging.debug(f"Menyimpan game term: '{w}'")
            result.append(w)
            continue

        # Pengecualian 2: Whitelist
        if w_lower in kata_id_pasti:
            if debug:
                logging.debug(f"Menyimpan kata whitelist: '{w}'")
            result.append(w)
            continue

        # Prioritas 1: Kata tidak relevan
        if w_lower in kata_tidak_relevan:
            if debug:
                logging.debug(f"Menghapus kata tidak relevan: '{w}'")
            stopword_removed.append(w)
            continue

        # Prioritas 2: Stopword Indonesia
        if w_lower in stop_words_id:
            if debug:
                logging.debug(f"Menghapus stopword Indonesia: '{w}'")
            stopword_removed.append(w)
            continue

        # Prioritas 3: Stopword Inggris
        if w_lower in stop_words_ing:
            if debug:
                logging.debug(f"Menghapus stopword Inggris: '{w}'")
            stopword_removed.append(w)
            continue

        # Jika lolos semua filter, simpan kata
        if debug:
            logging.debug(f"Menyimpan kata: '{w}'")
        result.append(w)

    if debug:
        logging.debug(f"Kata yang dihapus: {stopword_removed}")
        logging.debug(f"Kata setelah filter stopword: {result}")

    return result


def stemming_teks(words, debug=False):
    hasil = []
    stem_methods = []

    for w in words:
        wl = w.lower()
        wl_normalized = normalize_repeated_letters(wl)
        method_used = "original"

        if wl in stemming_dict:
            mapped = stemming_dict[wl]
            hasil.extend(mapped.split())
            method_used = "custom_dict"
        elif wl_normalized in stemming_dict:
            mapped = stemming_dict[wl_normalized]
            hasil.extend(mapped.split())
            method_used = "custom_dict_normalized"
        else:
            stemmed_sastrawi = cached_stemmer_stem(wl_normalized)
            if stemmed_sastrawi != wl_normalized:
                hasil.append(stemmed_sastrawi)
                method_used = "sastrawi"
            else:
                hasil.append(wl_normalized)
                method_used = "normalized"

        stem_methods.append(method_used)

    if debug and stem_methods:
        method_counts = Counter(stem_methods)
        logging.debug(f"Stemming methods digunakan: {dict(method_counts)}")

    return hasil


def deteksi_bukan_indonesia(text: str) -> bool:
    try:
        return cached_detect_language(text) != "id"
    except Exception:
        return False


def contains_foreign_words(text):
    """Mendeteksi apakah teks mengandung kata asing yang tidak ada di whitelist"""
    if not text or not isinstance(text, str):
        return False

    try:
        tokens = word_tokenize(text.lower())

        for token in tokens:
            if (
                len(token) <= 2
                or token in game_terms
                or token in kata_id_pasti
                or token in normalisasi_dict
                or token in stemming_dict
            ):
                continue

            try:
                lang = cached_detect_language(token)
                if lang != "id":
                    return True
            except Exception:
                continue

    except Exception as e:
        logging.error(f"Error deteksi kata asing: {e}")

    return False


# --- Fungsi preprocessing standar TANPA Token_Filtered di output ---
def proses_preprocessing_standar(teks, debug=False):
    """Fungsi preprocessing standar - Token_Filtered hanya di backend"""
    if pd.isna(teks) or not isinstance(teks, str) or not teks.strip():
        return ["", "", [], [], [], [], ""]  # 7 elemen tanpa Token_Filtered

    # Bersihkan teks
    clean = bersihkan_terjemahan(teks)
    folded = clean.lower()

    # Tokenisasi
    try:
        token = word_tokenize(folded)
    except Exception:
        token = word_pattern.findall(folded)

    if debug:
        logging.debug(f"Setelah tokenisasi: {token}")

    # Bersihkan token tidak diinginkan
    token_cleaned = bersihkan_token(token)

    if debug:
        logging.debug(f"Setelah bersihkan_token: {token_cleaned}")

    # **STEP TERSEMBUNYI: Hapus kata dari normalisasi_stopword_list.txt**
    # Step ini tetap dilakukan di backend tapi tidak ditampilkan di output
    token_filtered = hapus_kata_normalisasi_stopword(token_cleaned, debug)

    # Normalisasi teks - menggunakan token_filtered (hasil step tersembunyi)
    norm = normalisasi_teks(token_filtered) if token_filtered else []

    if debug:
        logging.debug(f"Setelah normalisasi: {norm}")

    # Stopword removal
    stop = hapus_stopword(norm, debug) if norm else []

    if debug:
        logging.debug(f"Setelah stopword removal: {stop}")

    # Stemming
    stem = stemming_teks(stop, debug) if stop else []

    if debug:
        logging.debug(f"Setelah stemming: {stem}")

    # Gabungkan hasil
    hasil = " ".join(stem) if stem else folded

    # **Kembalikan 7 elemen TANPA Token_Filtered**
    return [clean, folded, token, stop, norm, stem, hasil]


def proses_baris_standar(terjemahan, debug=False):
    try:
        if (
            pd.isna(terjemahan)
            or not isinstance(terjemahan, str)
            or not terjemahan.strip()
        ):
            clean_minimal = (
                whitespace_pattern.sub(" ", str(terjemahan)).strip()
                if isinstance(terjemahan, str)
                else ""
            )
            folded_minimal = clean_minimal.lower()
            return [clean_minimal, folded_minimal, [], [], [], [], folded_minimal]

        return proses_preprocessing_standar(terjemahan, debug)

    except Exception as e:
        logging.error(f"Error proses standar: {e}")
        clean = bersihkan_terjemahan(terjemahan) if isinstance(terjemahan, str) else ""
        folded = clean.lower() if clean else ""
        hasil = folded if folded else (clean if clean else str(terjemahan))
        return [clean, folded, [], [], [], [], hasil]


def proses_batch_standar(terjemahan_list, debug=False):
    """Proses batch data untuk preprocessing standar"""
    hasil = []
    for terjemahan in terjemahan_list:
        hasil.append(proses_baris_standar(terjemahan, debug))
    return hasil


# --- PERUBAHAN PENTING: Fungsi terjemahan ulang yang menerjemahkan hasil Fase 1 ---
def proses_terjemahan_ulang(teks_hasil_fase1, data_preprocessing_sebelumnya):
    """Melakukan terjemahan ulang pada hasil Fase 1 yang masih mengandung kata asing"""
    try:
        # **TERJEMAHKAN ULANG: teks_hasil_fase1 (bukan teks asli)**
        terjemahan_baru = translate_with_retry(teks_hasil_fase1)

        # Jika terjemahan baru error atau kosong, kembalikan hasil sebelumnya
        if not terjemahan_baru or is_error_message(terjemahan_baru):
            return data_preprocessing_sebelumnya

        # Gunakan fungsi preprocessing standar yang sama untuk hasil terjemahan baru
        return proses_preprocessing_standar(terjemahan_baru)

    except Exception as e:
        logging.error(f"Error terjemahan ulang setelah {MAX_RETRIES} percobaan: {e}")
        return data_preprocessing_sebelumnya


def save_processed_data(df, file_path=PREPRO_CSV_PATH):
    """Menyimpan data yang sudah diproses ke file CSV"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        logging.info(f"Data berhasil disimpan ke {file_path}")
        return True
    except Exception as e:
        logging.error(f"Gagal menyimpan data: {e}")
        return False


def load_processed_data(file_path=PREPRO_CSV_PATH):
    """Membaca data yang sudah diproses dari file CSV"""
    try:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            df = pd.read_csv(file_path)
            logging.info(f"Data berhasil dimuat dari {file_path}")

            # Konversi kolom string kembali ke list jika diperlukan
            list_columns = ["Tokenisasi", "Stopword", "Normalisasi", "Stemming"]
            for col in list_columns:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: x.split() if isinstance(x, str) and x != "" else []
                    )

            return df
        return None
    except Exception as e:
        logging.error(f"Gagal memuat data: {e}")
        return None


def save_csv(df: pd.DataFrame, file_path: str = PREPRO_CSV_PATH):
    """Menyimpan DataFrame ke CSV dengan format konsisten."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        df_for_file = df.copy(deep=True)

        # Konversi kolom list menjadi string yang aman untuk CSV
        list_cols = [
            "Tokenisasi",
            "Stopword",
            "Normalisasi",
            "Stemming",
        ]  # Token_Filtered dihapus
        for col in list_cols:
            if col in df_for_file.columns:
                df_for_file[col] = df_for_file[col].apply(
                    lambda val: " ".join(val) if isinstance(val, list) and val else ""
                )

        # Hapus kolom sensitif / tidak diperlukan
        df_for_file.drop(
            columns=[
                col
                for col in df_for_file.columns
                if col.lower() in {"ulasan", "terjemahan"}
            ],
            inplace=True,
            errors="ignore",
        )

        # Atur ulang kolom sesuai urutan yang diinginkan (tanpa Token_Filtered)
        kolom_urutan = [
            "SteamID",
            "Clean Data",
            "Case Folding",
            "Tokenisasi",
            "Stopword",  # Langsung dari Tokenisasi ke Stopword
            "Normalisasi",
            "Stemming",
            "Hasil",
            "Status",
        ]

        # Pastikan hanya ambil kolom yang memang ada
        kolom_ada = [col for col in kolom_urutan if col in df_for_file.columns]
        df_for_file = df_for_file[kolom_ada]

        # Simpan CSV dengan format yang aman untuk Windows
        with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
            # Tulis header terlebih dahulu
            f.write(",".join(['"' + col + '"' for col in df_for_file.columns]) + "\r\n")

            # Tulis data baris per baris
            for _, row in df_for_file.iterrows():
                row_data = []
                for value in row:
                    if pd.isna(value):
                        row_data.append('""')
                    elif isinstance(value, str):
                        escaped_value = value.replace('"', '""')
                        row_data.append('"' + escaped_value + '"')
                    else:
                        row_data.append('"' + str(value) + '"')
                f.write(",".join(row_data) + "\r\n")

        logging.info(f"File berhasil disimpan di {file_path}")
        return True
    except Exception as e:
        logging.error(f"Error save_csv: {e}")
        return False


# --- Routes ---
@prepro_bp.route("/preproses", methods=["POST"])
def preproses():
    try:
        file = request.files.get("file")
        if (
            not file
            or file.filename is None
            or not file.filename.lower().endswith(".csv")
        ):
            return jsonify(
                {"error": "Format file tidak valid, hanya mendukung CSV."}
            ), 400
        if file.content_length > MAX_FILE_SIZE:
            return jsonify({"error": "Ukuran file melebihi 2 MB."}), 400

        # Baca seluruh file
        df = pd.read_csv(file.stream)
        total_rows = len(df)

        if "Terjemahan" not in df.columns:
            return jsonify({"error": "Kolom 'Terjemahan' tidak ditemukan."}), 400

        df["Terjemahan"] = df["Terjemahan"].fillna("")

        # Fase 1: Preprocessing standar
        logging.info("Fase 1: Preprocessing standar...")
        with ProcessPoolExecutor(max_workers=4) as executor:
            chunks = [df[i : i + CHUNK_SIZE] for i in range(0, total_rows, CHUNK_SIZE)]
            futures = []

            for i, chunk in enumerate(chunks):
                futures.append(
                    executor.submit(proses_batch_standar, chunk["Terjemahan"].tolist())
                )

            processed_chunks = []
            for i, future in enumerate(as_completed(futures)):
                logging.info(f"Chunk {i + 1}: Memproses {len(chunks[i])} baris...")
                hasil_list = future.result()
                # **Hanya 7 kolom tanpa Token_Filtered**
                hasil_df = pd.DataFrame(
                    hasil_list,
                    columns=[
                        "Clean Data",
                        "Case Folding",
                        "Tokenisasi",
                        "Stopword",
                        "Normalisasi",
                        "Stemming",
                        "Hasil",
                    ],
                )

                chunk_df = chunks[i].reset_index(drop=True)
                chunk_df = pd.concat([chunk_df, hasil_df], axis=1)
                if "Status" not in chunk_df.columns:
                    chunk_df["Status"] = ""
                processed_chunks.append(chunk_df)

        result_df = pd.concat(processed_chunks, ignore_index=True)

        # Fase 2: Terjemahan ulang untuk baris yang membutuhkan
        logging.info("Fase 2: Terjemahan ulang...")
        baris_diperbaiki = 0

        for idx in range(len(result_df)):
            if idx % 100 == 0:
                logging.info(f"Memeriksa baris {idx + 1} dari {len(result_df)}")

            # Ambil hasil dari Fase 1
            hasil_sebelumnya = result_df.at[idx, "Hasil"]

            # Deteksi apakah hasil_sebelumnya masih mengandung kata asing
            if not contains_foreign_words(hasil_sebelumnya):
                # Jika tidak ada kata asing, pertahankan hasil Fase 1
                continue

            # Jika masih ada kata asing, siapkan data preprocessing sebelumnya untuk fallback
            data_preprocessing_sebelumnya = [
                result_df.at[idx, "Clean Data"],
                result_df.at[idx, "Case Folding"],
                result_df.at[idx, "Tokenisasi"],
                result_df.at[idx, "Stopword"],
                result_df.at[idx, "Normalisasi"],
                result_df.at[idx, "Stemming"],
                hasil_sebelumnya,
            ]

            # **PERUBAHAN PENTING: Terjemahkan ulang hasil_sebelumnya (hasil Fase 1)**
            hasil_baru = proses_terjemahan_ulang(
                hasil_sebelumnya, data_preprocessing_sebelumnya
            )

            # Jika hasil baru berbeda dari sebelumnya, update semua kolom
            if hasil_baru[6] != hasil_sebelumnya:  # Index 6 adalah "Hasil"
                # Update semua kolom dengan hasil baru
                result_df.at[idx, "Clean Data"] = hasil_baru[0]
                result_df.at[idx, "Case Folding"] = hasil_baru[1]
                result_df.at[idx, "Tokenisasi"] = hasil_baru[2]
                result_df.at[idx, "Stopword"] = hasil_baru[3]
                result_df.at[idx, "Normalisasi"] = hasil_baru[4]
                result_df.at[idx, "Stemming"] = hasil_baru[5]
                result_df.at[idx, "Hasil"] = hasil_baru[6]
                baris_diperbaiki += 1

        logging.info(f"Terjemahan ulang selesai. {baris_diperbaiki} baris diperbaiki.")

        # Simpan CSV otomatis
        save_csv(result_df)
        simpan_cache_ke_file()
        simpan_cache_bahasa()

        # Hitung statistik
        total_baris = len(result_df)
        baris_kosong = len(result_df[result_df["Hasil"].str.strip() == ""])
        persentase_kosong = (baris_kosong / total_baris * 100) if total_baris > 0 else 0

        return jsonify(
            {
                "message": f"Preprocessing selesai dan disimpan! {baris_diperbaiki} baris diterjemahkan ulang, {baris_kosong} baris kosong ({persentase_kosong:.2f}%)",
                "data": result_df.to_dict(orient="records"),
                "stats": {
                    "total": total_baris,
                    "diperbaiki": baris_diperbaiki,
                    "kosong": baris_kosong,
                    "persentase_kosong": persentase_kosong,
                },
            }
        )
    except Exception as e:
        logging.error(f"Error dalam preprocessing: {e}")
        return jsonify({"error": str(e)}), 500


@prepro_bp.route("/save_csv", methods=["POST"])
def save_csv_manual():
    """Menyimpan CSV secara manual dari data JSON yang dikirim frontend."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Tidak ada data yang dikirim."}), 400

        df = pd.DataFrame(data)

        if save_csv(df):
            return send_file(
                PREPRO_CSV_PATH,
                as_attachment=True,
                download_name="processed_data.csv",
                mimetype="text/csv",
            )
        else:
            return jsonify({"error": "Gagal menyimpan CSV."}), 500
    except Exception as e:
        logging.error(f"Error save_csv_manual: {e}")
        return jsonify({"error": str(e)}), 500


@prepro_bp.route("/download_csv", methods=["GET"])
def download_csv():
    """Mengunduh file hasil preprocessing CSV secara manual."""
    if os.path.exists(PREPRO_CSV_PATH):
        return send_file(
            PREPRO_CSV_PATH,
            as_attachment=True,
            download_name="processed_data.csv",
            mimetype="text/csv",
        )
    return jsonify({"error": "File hasil preprocessing tidak ditemukan."}), 404


@prepro_bp.route("/debug_prepro", methods=["POST"])
def debug_prepro():
    """Endpoint untuk debugging preprocessing pada teks tertentu"""
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Tidak ada teks yang dikirim."}), 400

        teks = data["text"]
        logging.info(f"Debug: Memproses teks: {teks}")

        # Lakukan preprocessing dengan debug mode
        hasil = proses_preprocessing_standar(teks, debug=True)

        return jsonify(
            {
                "clean": hasil[0],
                "folded": hasil[1],
                "tokens": hasil[2],
                "stopwords_removed": hasil[3],
                "normalized": hasil[4],
                "stemmed": hasil[5],
                "final": hasil[6],
            }
        )
    except Exception as e:
        logging.error(f"Error debug_prepro: {e}")
        return jsonify({"error": str(e)}), 500


@prepro_bp.route("/load_data", methods=["GET"])
def load_data():
    """Memuat data dari file CSV saat halaman dibuka"""
    try:
        df = load_processed_data()
        if df is not None and not df.empty:
            # Hitung statistik
            total_baris = len(df)
            baris_kosong = len(df[df["Hasil"].str.strip() == ""])
            persentase_kosong = (
                (baris_kosong / total_baris * 100) if total_baris > 0 else 0
            )

            # Konversi DataFrame ke dictionary
            data = df.to_dict(orient="records")

            return jsonify(
                {
                    "status": "success",
                    "data": data,
                    "stats": {
                        "total": total_baris,
                        "kosong": baris_kosong,
                        "persentase_kosong": persentase_kosong,
                    },
                }
            )

        return jsonify(
            {"status": "empty", "data": [], "message": "Tidak ada data tersimpan"}
        )

    except Exception as e:
        logging.error(f"Error load_data: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@prepro_bp.route("/delete_csv", methods=["POST"])
def delete_csv():
    """Menghapus file CSV yang sudah diproses dari server"""
    try:
        if os.path.exists(PREPRO_CSV_PATH):
            os.remove(PREPRO_CSV_PATH)
            logging.info(f"File {PREPRO_CSV_PATH} berhasil dihapus")
            return jsonify(
                {
                    "status": "success",
                    "message": "File CSV berhasil dihapus dari server",
                }
            )
        else:
            return jsonify(
                {"status": "error", "message": "File CSV tidak ditemukan"}
            ), 404
    except Exception as e:
        logging.error(f"Error deleting CSV file: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@prepro_bp.route("/get_csv_data", methods=["GET"])
def get_csv_data():
    """Membaca data dari file CSV yang sudah diproses"""
    try:
        if not os.path.exists(PREPRO_CSV_PATH):
            return jsonify({"exists": False, "message": "File CSV tidak ditemukan"})

        # Baca file CSV
        df = pd.read_csv(PREPRO_CSV_PATH)

        # Konversi kolom string kembali ke list jika diperlukan
        list_columns = ["Tokenisasi", "Stopword", "Normalisasi", "Stemming"]
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: x.split() if isinstance(x, str) and x != "" else []
                )

        # Konversi ke dictionary
        data = df.to_dict(orient="records")

        return jsonify(
            {"exists": True, "data": data, "message": "Data berhasil dibaca dari CSV"}
        )
    except Exception as e:
        logging.error(f"Error get_csv_data: {e}")
        return jsonify({"exists": False, "error": str(e)})


@prepro_bp.route("/debug_stopword", methods=["GET"])
def debug_stopword():
    """Endpoint untuk debugging stopword"""
    try:
        # Test dengan contoh teks
        test_text = "yang dan di dari ke pada ini itu dengan untuk"
        hasil = proses_preprocessing_standar(test_text, debug=True)

        return jsonify(
            {
                "status": "success",
                "test_text": test_text,
                "result": hasil[6],  # Hasil akhir
                "stopwords_loaded": {
                    "indonesia_count": len(stop_words_id),
                    "english_count": len(stop_words_ing),
                    "sample_id": list(stop_words_id)[:10],
                    "sample_ing": list(stop_words_ing)[:10],
                },
            }
        )
    except Exception as e:
        logging.error(f"Error debug_stopword: {e}")
        return jsonify({"error": str(e)}), 500


@prepro_bp.route("/")
def index():
    return render_template("preprosessing.html", page_name="prepro"), 200

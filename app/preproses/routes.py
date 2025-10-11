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

# Path cache files
CACHE_FOLDER = os.path.join(BASE_DIR, "uploads", "preproses")
CACHE_FILE = os.path.join(CACHE_FOLDER, "cache_translate.json")
LANGUAGE_CACHE_FILE = os.path.join(CACHE_FOLDER, "cache_language.json")
TRANSLATION_CACHE_FILE = os.path.join(
    BASE_DIR, "uploads", "terjemahan", "translation_cache.json"
)


# --- Fungsi helper untuk cache ---
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


def simpan_cache_bahasa():
    try:
        os.makedirs(os.path.dirname(LANGUAGE_CACHE_FILE), exist_ok=True)
        with open(LANGUAGE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(language_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Error simpan cache bahasa: {e}")


def simpan_cache_ke_file():
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(terjemahan_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Error simpan cache: {e}")


# --- Cache untuk deteksi bahasa ---
language_cache = {}
if os.path.exists(LANGUAGE_CACHE_FILE):
    try:
        with open(LANGUAGE_CACHE_FILE, "r", encoding="utf-8") as f:
            language_cache = json.load(f)
    except Exception:
        language_cache = {}

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


# --- FUNGSI LOADING NORMALISASI YANG DIPERBAIKI ---
def load_normalization_dict():
    """Fungsi loading yang lebih robust untuk normalisasi_list.txt"""
    global normalisasi_dict
    normalisasi_dict.clear()

    normalisasi_file = os.path.join(TXT_DIR, "normalisasi_list.txt")
    if not os.path.exists(normalisasi_file):
        logging.error("❌ File normalisasi_list.txt tidak ditemukan!")
        return

    loaded_count = 0
    error_count = 0

    try:
        with open(normalisasi_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip baris kosong dan komentar
                if not line or line.startswith("#"):
                    continue

                # CARI POSISI "=" PERTAMA - lebih fleksibel
                equal_pos = line.find("=")
                if equal_pos == -1:
                    logging.warning(
                        f"Baris {line_num}: Format tidak valid (tidak ada '='): {line}"
                    )
                    error_count += 1
                    continue

                # Split berdasarkan posisi =
                k = line[:equal_pos].strip().lower()
                v = line[equal_pos + 1 :].strip().lower()

                # Validasi
                if not k:
                    logging.warning(f"Baris {line_num}: Key kosong: {line}")
                    error_count += 1
                    continue
                if not v:
                    logging.warning(f"Baris {line_num}: Value kosong: {line}")
                    error_count += 1
                    continue

                # Simpan ke dictionary
                normalisasi_dict[k] = v
                loaded_count += 1

                # Juga tambahkan variasi tanpa spasi jika ada spasi
                if " " in k:
                    k_no_space = k.replace(" ", "")
                    if k_no_space != k and k_no_space not in normalisasi_dict:
                        normalisasi_dict[k_no_space] = v
                        loaded_count += 1

        logging.info(
            f"✅ SUCCESS: Loaded {loaded_count} entries dari normalisasi_list.txt, {error_count} errors"
        )

        # Debug: tampilkan beberapa entri untuk verifikasi
        sample_keys = ["bgtt", "bs", "td", "klo", "yg", "dgn", "gue", "lo"]
        found_keys = []
        missing_keys = []

        for key in sample_keys:
            if key in normalisasi_dict:
                found_keys.append(f"'{key}' -> '{normalisasi_dict[key]}'")
            else:
                missing_keys.append(key)

        if found_keys:
            logging.info(f"✅ Sample entries: {', '.join(found_keys)}")
        if missing_keys:
            logging.warning(f"❌ Missing keys: {missing_keys}")

    except Exception as e:
        logging.error(f"❌ Error load_normalization_dict: {e}")


# --- Fungsi untuk memuat cache terjemahan yang sudah ada ---
def load_translation_cache():
    """Memuat cache terjemahan dari folder terjemahan yang sudah ada"""
    translation_cache = {}
    try:
        if os.path.exists(TRANSLATION_CACHE_FILE):
            with open(TRANSLATION_CACHE_FILE, "r", encoding="utf-8") as f:
                translation_cache = json.load(f)
            logging.info(
                f"✅ Loaded {len(translation_cache)} entries dari translation_cache.json"
            )

            # Validasi dan bersihkan cache
            keys_to_remove = []
            for key, value in translation_cache.items():
                if is_error_message(value) or is_error_message(key):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del translation_cache[key]

            if keys_to_remove:
                logging.info(
                    f"Menghapus {len(keys_to_remove)} entri error dari translation cache"
                )
        else:
            logging.warning(
                "File translation_cache.json tidak ditemukan di folder terjemahan"
            )
    except Exception as e:
        logging.error(f"Error load_translation_cache: {e}")

    return translation_cache


# --- Fungsi untuk sinkronisasi cache antara preprocessing dan terjemahan ---
def sync_translation_caches():
    """Menyinkronkan cache antara preprocessing dan terjemahan"""
    try:
        # Load cache dari terjemahan
        translation_cache = load_translation_cache()

        # Gabungkan cache (prioritaskan cache terjemahan yang sudah ada)
        updated_count = 0
        for key, value in translation_cache.items():
            if key not in terjemahan_cache:
                terjemahan_cache[key] = value
                updated_count += 1

        if updated_count > 0:
            logging.info(
                f"✅ Sinkronisasi cache: {updated_count} entries baru dari terjemahan"
            )
            # Simpan cache gabungan
            simpan_cache_ke_file()

    except Exception as e:
        logging.error(f"Error sync_translation_caches: {e}")


# --- Fungsi inisialisasi sekali pakai ---
def initialize_preprocessing_data():
    """Fungsi untuk inisialisasi data preprocessing sekali saja"""
    global _INITIALIZED, stop_words_id, stop_words_ing, normalisasi_dict, stemming_dict
    global game_terms, kata_tidak_relevan, kata_id_pasti

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

    # PERBAIKAN: Load normalisasi dictionary dengan fungsi baru
    load_normalization_dict()

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


def translate_with_retry(text, source="auto", target="id", max_len=5000):
    """Fungsi terjemahan dengan mekanisme retry dan multiple cache sources"""

    # 1. Cek di cache preprocessing terlebih dahulu
    if text in terjemahan_cache:
        logging.debug(f"✅ Menggunakan cache preprocessing untuk: {text[:50]}...")
        return terjemahan_cache[text]

    # 2. Cek di cache terjemahan yang sudah ada
    translation_existing_cache = load_translation_cache()
    if text in translation_existing_cache:
        logging.debug(f"✅ Menggunakan cache terjemahan existing untuk: {text[:50]}...")
        # Simpan juga ke cache preprocessing untuk akses future
        terjemahan_cache[text] = translation_existing_cache[text]
        simpan_cache_ke_file()
        return translation_existing_cache[text]

    # 3. Jika tidak ada di cache, lakukan terjemahan
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

            hasil_terjemahan = " ".join(hasil)

            # Simpan ke cache preprocessing
            terjemahan_cache[text] = hasil_terjemahan
            simpan_cache_ke_file()

            return hasil_terjemahan

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


# --- FUNGSI NORMALISASI YANG DIPERBAIKI ---
def normalisasi_teks(words, debug=False):
    """Fungsi normalisasi yang lebih robust dengan multiple fallback"""
    hasil = []
    normalisasi_count = 0
    skipped_words = []

    for w in words:
        wl = w.lower().strip()

        # Skip jika kata kosong
        if not wl:
            continue

        # CLEANING LEBIH AGGRESIF - hapus karakter non-alphabet di awal/akhir
        wl_clean = re.sub(r"^[^a-z]+|[^a-z]+$", "", wl)

        # PRIORITAS 1: Cek kata bersih (setelah cleaning)
        if wl_clean and wl_clean in normalisasi_dict:
            mapped = normalisasi_dict[wl_clean]
            if debug:
                logging.debug(
                    f"🔧 Normalisasi (clean): '{wl}' -> '{wl_clean}' -> '{mapped}'"
                )
            hasil.extend(mapped.split())
            normalisasi_count += 1
            continue

        # PRIORITAS 2: Cek kata asli (tanpa cleaning)
        if wl in normalisasi_dict:
            mapped = normalisasi_dict[wl]
            if debug:
                logging.debug(f"🔧 Normalisasi (original): '{wl}' -> '{mapped}'")
            hasil.extend(mapped.split())
            normalisasi_count += 1
            continue

        # PRIORITAS 3: Cek variasi dengan menghilangkan pengulangan huruf
        wl_normalized = normalize_repeated_letters(wl)
        if wl_normalized != wl and wl_normalized in normalisasi_dict:
            mapped = normalisasi_dict[wl_normalized]
            if debug:
                logging.debug(
                    f"🔧 Normalisasi (repeated): '{wl}' -> '{wl_normalized}' -> '{mapped}'"
                )
            hasil.extend(mapped.split())
            normalisasi_count += 1
            continue

        # Jika tidak ada di dictionary, pertahankan kata asli
        hasil.append(wl)
        skipped_words.append(wl)

    if debug:
        if normalisasi_count > 0:
            logging.info(f"✅ Dilakukan {normalisasi_count} normalisasi")
        if skipped_words:
            logging.info(f"❌ Kata tidak ternormalisasi: {skipped_words}")

    return hasil


# --- FUNGSI BERSIHKAN TOKEN YANG DIPERBAIKI ---
def bersihkan_token(tokens, debug=False):
    """Membersihkan token-token yang tidak diinginkan - versi diperbaiki"""
    hasil = []
    skipped_count = 0

    for token in tokens:
        normalized_token = normalize_repeated_letters(token)

        # PRIORITAS: JANGAN hapus token yang ada di normalisasi_dict
        if normalized_token in normalisasi_dict:
            if debug:
                logging.debug(
                    f"🔧 Pertahankan token untuk normalisasi: '{normalized_token}'"
                )
            hasil.append(normalized_token)
            continue

        # JANGAN hapus token yang merupakan hasil normalisasi
        is_normalized_result = any(
            normalized_token in value.split() for value in normalisasi_dict.values()
        )
        if is_normalized_result:
            if debug:
                logging.debug(f"🔧 Pertahankan hasil normalisasi: '{normalized_token}'")
            hasil.append(normalized_token)
            continue

        # Filter token yang tidak diinginkan
        if (
            number_unit_pattern.match(normalized_token)
            or short_word_pattern.match(normalized_token)
            or mixed_alnum_pattern.match(normalized_token)
        ):
            if debug:
                logging.debug(f"🗑️ Hapus token tidak diinginkan: '{normalized_token}'")
            skipped_count += 1
            continue

        if normalized_token.isdigit():
            if debug:
                logging.debug(f"🗑️ Hapus token angka: '{normalized_token}'")
            skipped_count += 1
            continue

        hasil.append(normalized_token)

    if debug and skipped_count > 0:
        logging.info(f"🗑️ Dihapus {skipped_count} token tidak diinginkan")

    return hasil


def hapus_stopword(words, debug=False):
    """Fungsi stopword removal yang diperbaiki"""
    if not words:
        return []

    if debug:
        logging.debug(f"Kata sebelum filter stopword: {words}")

    result = []
    stopword_removed = []

    for w in words:
        w_lower = w.lower()

        # Pengecualian 1: Game terms
        if w_lower in game_terms:
            if debug:
                logging.debug(f"🎮 Menyimpan game term: '{w}'")
            result.append(w)
            continue

        # Pengecualian 2: Whitelist
        if w_lower in kata_id_pasti:
            if debug:
                logging.debug(f"📝 Menyimpan kata whitelist: '{w}'")
            result.append(w)
            continue

        # Prioritas 1: Kata tidak relevan
        if w_lower in kata_tidak_relevan:
            if debug:
                logging.debug(f"🗑️ Menghapus kata tidak relevan: '{w}'")
            stopword_removed.append(w)
            continue

        # Prioritas 2: Stopword Indonesia
        if w_lower in stop_words_id:
            if debug:
                logging.debug(f"🗑️ Menghapus stopword Indonesia: '{w}'")
            stopword_removed.append(w)
            continue

        # Prioritas 3: Stopword Inggris
        if w_lower in stop_words_ing:
            if debug:
                logging.debug(f"🗑️ Menghapus stopword Inggris: '{w}'")
            stopword_removed.append(w)
            continue

        # Jika lolos semua filter, simpan kata
        if debug:
            logging.debug(f"💾 Menyimpan kata: '{w}'")
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


# --- FUNGSI DETEKSI BAHASA ASING YANG DIPERBAIKI ---
def contains_foreign_words(text):
    """Mendeteksi apakah teks mengandung kata asing yang tidak ada di whitelist - VERSI DIPERBAIKI"""
    if not text or not isinstance(text, str):
        return False

    try:
        tokens = word_tokenize(text.lower())
        foreign_words_found = []

        for token in tokens:
            # KECUALIKAN: kata pendek, game terms, whitelist, dan kata yang sudah dinormalisasi/distem
            if (
                len(token) <= 2  # Kata pendek diabaikan (accuracy rendah)
                or token in game_terms  # Game terms dipertahankan
                or token in kata_id_pasti  # Kata whitelist dipertahankan
                or token in normalisasi_dict  # Kata yang sudah dinormalisasi
                or token in stemming_dict  # Kata yang sudah distem
            ):
                continue  # Skip deteksi bahasa untuk token ini

            # Hanya token yang TIDAK termasuk kategori di atas yang dicek bahasa
            try:
                lang = cached_detect_language(token)
                if lang != "id":
                    foreign_words_found.append((token, lang))
            except Exception:
                continue

        # Jika ditemukan kata asing, log dan return True
        if foreign_words_found:
            logging.debug(f"🌍 Kata asing terdeteksi: {foreign_words_found}")
            return True

    except Exception as e:
        logging.error(f"Error deteksi kata asing: {e}")

    return False


# --- FUNGSI PREPROCESSING STANDAR DENGAN PERBAIKAN URUTAN ---
def proses_preprocessing_standar(teks, debug=False):
    """Fungsi preprocessing standar - dengan urutan yang diperbaiki dan logging"""
    if pd.isna(teks) or not isinstance(teks, str) or not teks.strip():
        return ["", "", [], [], [], [], ""]  # 7 elemen

    # Bersihkan teks
    clean = bersihkan_terjemahan(teks)
    folded = clean.lower()

    if debug:
        logging.info(f"📥 Input: '{teks}'")
        logging.info(f"🧹 Clean: '{clean}'")
        logging.info(f"🔠 Case Folding: '{folded}'")

    # Tokenisasi
    try:
        token = word_tokenize(folded)
    except Exception:
        token = word_pattern.findall(folded)

    if debug:
        logging.info(f"🔪 Tokenisasi: {token}")

    # PERBAIKAN URUTAN: Normalisasi dilakukan SEBELUM pembersihan token
    norm = normalisasi_teks(token, debug) if token else []

    if debug:
        logging.info(f"🔧 Setelah normalisasi: {norm}")

    # Bersihkan token tidak diinginkan (setelah normalisasi)
    token_cleaned = bersihkan_token(norm, debug) if norm else []

    if debug:
        logging.info(f"🧽 Setelah bersihkan_token: {token_cleaned}")

    # Stopword removal
    stop = hapus_stopword(token_cleaned, debug) if token_cleaned else []

    if debug:
        logging.info(f"🚫 Setelah stopword removal: {stop}")

    # Stemming
    stem = stemming_teks(stop, debug) if stop else []

    if debug:
        logging.info(f"✂️ Setelah stemming: {stem}")

    # Gabungkan hasil
    hasil = " ".join(stem) if stem else folded

    if debug:
        logging.info(f"🎯 Hasil akhir: '{hasil}'")

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


# --- Fungsi terjemahan ulang yang menerjemahkan hasil Fase 1 ---
def proses_terjemahan_ulang(teks_hasil_fase1, data_preprocessing_sebelumnya):
    """Melakukan terjemahan ulang pada hasil Fase 1 yang masih mengandung kata asing"""
    try:
        # TERJEMAHKAN ULANG: teks_hasil_fase1 (bukan teks asli)
        terjemahan_baru = translate_with_retry(teks_hasil_fase1)

        # Jika terjemahan baru error atau kosong, kembalikan hasil sebelumnya
        if not terjemahan_baru or is_error_message(terjemahan_baru):
            return data_preprocessing_sebelumnya

        # PERBAIKAN: Tambahkan debug=False saat memanggil proses_preprocessing_standar
        return proses_preprocessing_standar(terjemahan_baru, debug=False)

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
        ]
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

        # Atur ulang kolom sesuai urutan yang diinginkan
        kolom_urutan = [
            "SteamID",
            "Clean Data",
            "Case Folding",
            "Tokenisasi",
            "Stopword",
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


# --- ENDPOINT BARU UNTUK TESTING NORMALISASI ---
@prepro_bp.route("/test_normalization", methods=["POST"])
def test_normalization():
    """Test normalisasi dengan teks contoh"""
    try:
        data = request.get_json()
        test_text = data.get("text", "gue mau makan bgtt ntar klo bs")

        # Paksa reload dictionary
        load_normalization_dict()

        # Test normalisasi
        tokens = word_tokenize(test_text.lower())
        normalized = normalisasi_teks(tokens, debug=True)

        # Analisis detail
        analysis = []
        for token in tokens:
            in_dict = token in normalisasi_dict
            mapped_to = normalisasi_dict.get(token, "TIDAK_ADA")
            analysis.append(
                {
                    "token": token,
                    "in_dictionary": in_dict,
                    "mapped_to": mapped_to,
                    "status": "✅ NORMALIZED" if in_dict else "❌ MISSING",
                }
            )

        return jsonify(
            {
                "input_text": test_text,
                "tokens": tokens,
                "normalized_result": normalized,
                "analysis": analysis,
                "dictionary_sample": dict(list(normalisasi_dict.items())[:10]),
                "test_keys_presence": {
                    "bgtt": "bgtt" in normalisasi_dict,
                    "bs": "bs" in normalisasi_dict,
                    "td": "td" in normalisasi_dict,
                    "klo": "klo" in normalisasi_dict,
                    "yg": "yg" in normalisasi_dict,
                    "dgn": "dgn" in normalisasi_dict,
                    "gue": "gue" in normalisasi_dict,
                    "lo": "lo" in normalisasi_dict,
                },
            }
        )

    except Exception as e:
        logging.error(f"Error test_normalization: {e}")
        return jsonify({"error": str(e)}), 500


@prepro_bp.route("/validate_dictionary", methods=["GET"])
def validate_dictionary():
    """Validasi isi kamus normalisasi"""
    sample_size = min(20, len(normalisasi_dict))
    sample_items = dict(list(normalisasi_dict.items())[:sample_size])

    # Cek contoh kata yang seharusnya dinormalisasi
    test_cases = ["gue", "lo", "bgtt", "bs", "td", "klo", "yg", "dgn", "ntar", "ga"]
    test_results = {}

    for test_word in test_cases:
        test_results[test_word] = {
            "in_dict": test_word in normalisasi_dict,
            "mapped_to": normalisasi_dict.get(test_word, "TIDAK_ADA"),
        }

    return jsonify(
        {
            "dictionary_size": len(normalisasi_dict),
            "sample_items": sample_items,
            "test_cases": test_results,
            "common_missing": [k for k in test_cases if k not in normalisasi_dict],
        }
    )


# --- Routes yang diperbaiki ---
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

        # Sinkronisasi cache sebelum memulai preprocessing
        sync_translation_caches()

        # Fase 1: Preprocessing standar dengan urutan yang diperbaiki
        logging.info("Fase 1: Preprocessing standar (dengan urutan diperbaiki)...")
        with ProcessPoolExecutor(max_workers=4) as executor:
            chunks = [df[i : i + CHUNK_SIZE] for i in range(0, total_rows, CHUNK_SIZE)]
            futures = []

            for i, chunk in enumerate(chunks):
                futures.append(
                    executor.submit(
                        proses_batch_standar, chunk["Terjemahan"].tolist(), False
                    )
                )

            processed_chunks = []
            for i, future in enumerate(as_completed(futures)):
                logging.info(f"Chunk {i + 1}: Memproses {len(chunks[i])} baris...")
                hasil_list = future.result()
                # 7 kolom dengan urutan yang diperbaiki
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

            # Terjemahkan ulang hasil_sebelumnya (hasil Fase 1)
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


@prepro_bp.route("/debug_normalization", methods=["POST"])
def debug_normalization():
    """Endpoint khusus untuk debugging normalisasi"""
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Tidak ada teks yang dikirim."}), 400

        teks = data["text"]
        logging.info(f"Debug Normalization: Memproses teks: {teks}")

        # Tokenisasi saja dulu
        folded = teks.lower()
        tokens = word_tokenize(folded)

        # Cek normalisasi untuk setiap token
        normalization_results = []
        for token in tokens:
            token_lower = token.lower()
            is_in_dict = token_lower in normalisasi_dict
            normalized_value = normalisasi_dict.get(token_lower, "TIDAK_ADA")
            normalization_results.append(
                {
                    "token": token,
                    "lowercase": token_lower,
                    "in_normalization_dict": is_in_dict,
                    "normalized_to": normalized_value,
                }
            )

        # Proses lengkap untuk perbandingan
        hasil_lengkap = proses_preprocessing_standar(teks, debug=True)

        return jsonify(
            {
                "input_text": teks,
                "tokens": tokens,
                "normalization_check": normalization_results,
                "full_processing": {
                    "clean": hasil_lengkap[0],
                    "folded": hasil_lengkap[1],
                    "tokens": hasil_lengkap[2],
                    "stopwords_removed": hasil_lengkap[3],
                    "normalized": hasil_lengkap[4],
                    "stemmed": hasil_lengkap[5],
                    "final": hasil_lengkap[6],
                },
                "normalization_dict_sample": dict(list(normalisasi_dict.items())[:10]),
            }
        )
    except Exception as e:
        logging.error(f"Error debug_normalization: {e}")
        return jsonify({"error": str(e)}), 500


@prepro_bp.route("/clear_cache", methods=["POST"])
def clear_cache():
    """Membersihkan cache terjemahan dan bahasa"""
    try:
        terjemahan_cache.clear()
        language_cache.clear()

        # Hapus file cache
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        if os.path.exists(LANGUAGE_CACHE_FILE):
            os.remove(LANGUAGE_CACHE_FILE)

        # Re-initialize data
        global _INITIALIZED
        _INITIALIZED = False
        initialize_preprocessing_data()

        return jsonify({"status": "success", "message": "Cache berhasil dibersihkan"})
    except Exception as e:
        logging.error(f"Error clear_cache: {e}")
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


# --- ENDPOINT DEBUG BARU UNTUK ANALISIS MASALAH ---
@prepro_bp.route("/analyze_issues", methods=["POST"])
def analyze_issues():
    """Endpoint untuk menganalisis masalah normalisasi dan kata asing"""
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Tidak ada teks yang dikirim."}), 400

        teks = data["text"]
        logging.info(f"🔍 Analisis masalah untuk: {teks}")

        # Proses dengan debug mode
        hasil = proses_preprocessing_standar(teks, debug=True)

        # Analisis tambahan
        tokens = word_tokenize(teks.lower())
        normalization_analysis = []
        foreign_analysis = []

        for token in tokens:
            # Analisis normalisasi
            if token in normalisasi_dict:
                normalization_analysis.append(
                    {
                        "token": token,
                        "in_dict": True,
                        "normalized_to": normalisasi_dict[token],
                        "status": "✅ BISA DINORMALISASI",
                    }
                )
            else:
                normalization_analysis.append(
                    {
                        "token": token,
                        "in_dict": False,
                        "normalized_to": None,
                        "status": "❌ TIDAK ADA DI DICTIONARY",
                    }
                )

            # Analisis kata asing
            if (
                len(token) > 2
                and token not in game_terms
                and token not in kata_id_pasti
                and token not in normalisasi_dict
                and token not in stemming_dict
            ):
                try:
                    lang = cached_detect_language(token)
                    if lang != "id":
                        foreign_analysis.append(
                            {
                                "token": token,
                                "language": lang,
                                "status": "🌍 BAHASA ASING",
                            }
                        )
                except Exception:
                    pass

        return jsonify(
            {
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
                "normalization_analysis": normalization_analysis,
                "foreign_words_analysis": foreign_analysis,
                "dictionary_info": {
                    "normalization_dict_size": len(normalisasi_dict),
                    "game_terms_size": len(game_terms),
                    "whitelist_size": len(kata_id_pasti),
                    "sample_normalization": dict(list(normalisasi_dict.items())[:10]),
                },
            }
        )
    except Exception as e:
        logging.error(f"Error analyze_issues: {e}")
        return jsonify({"error": str(e)}), 500


@prepro_bp.route("/check_dictionary", methods=["POST"])
def check_dictionary():
    """Endpoint untuk mengecek apakah kata ada di dictionary"""
    try:
        data = request.get_json()
        if not data or "word" not in data:
            return jsonify({"error": "Tidak ada kata yang dikirim."}), 400

        word = data["word"].lower()

        in_normalization = word in normalisasi_dict
        in_stemming = word in stemming_dict
        in_game_terms = word in game_terms
        in_whitelist = word in kata_id_pasti

        return jsonify(
            {
                "word": word,
                "in_normalization_dict": in_normalization,
                "normalization_value": normalisasi_dict.get(word)
                if in_normalization
                else None,
                "in_stemming_dict": in_stemming,
                "stemming_value": stemming_dict.get(word) if in_stemming else None,
                "in_game_terms": in_game_terms,
                "in_whitelist": in_whitelist,
            }
        )
    except Exception as e:
        logging.error(f"Error check_dictionary: {e}")
        return jsonify({"error": str(e)}), 500


@prepro_bp.route("/")
def index():
    return render_template("preprosessing.html", page_name="prepro"), 200

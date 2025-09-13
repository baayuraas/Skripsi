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

# --- Konfigurasi dasar ---
prepro_bp = Blueprint(
    "prepro",
    __name__,
    url_prefix="/prepro",
    template_folder="templates",
    static_folder="static",
)

# Pastikan hasil deteksi bahasa konsisten
DetectorFactory.seed = 0

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "preproses")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
PREPRO_CSV_PATH = os.path.join(UPLOAD_FOLDER, "processed_data.csv")
TXT_DIR = os.path.dirname(os.path.abspath(__file__))

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

# Validasi file yang diperlukan
for file_name in required_txt_files:
    file_path = os.path.join(TXT_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"[WARNING] File {file_name} tidak ditemukan di {TXT_DIR}")
    else:
        print(f"[INFO] File {file_name} ditemukan")


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
        print(f"[STOPWORD] Loaded {len(stopwords_set)} kata dari {filename}")
    except Exception as e:
        print(f"[ERROR load_stopwords {filename}]: {e}")
    return stopwords_set


# Inisialisasi stopword factory
stopword_factory = StopWordRemoverFactory()
stemmer = StemmerFactory().create_stemmer()

# Load stopword dasar Indonesia
stop_words_id = set(stopword_factory.get_stop_words()).union(
    set(stopwords.words("indonesian"))
)

# Load stopword Indonesia dari file
id_stopwords = load_stopwords(os.path.join(TXT_DIR, "stopword_list.txt"))
stop_words_id.update(id_stopwords)

# Load stopword Inggris dari file
stopword_ing_file = os.path.join(TXT_DIR, "stopword_list_ing.txt")
stop_words_ing = set()
if os.path.exists(stopword_ing_file):
    stop_words_ing = load_stopwords(stopword_ing_file)
else:
    # Fallback ke stopword Inggris dari NLTK jika file tidak ada
    stop_words_ing = set(stopwords.words("english"))
    print("[STOPWORD ING] Using NLTK English stopwords as fallback")

print(f"[STOPWORD] Total stopwords Indonesia: {len(stop_words_id)}")
print(f"[STOPWORD] Total stopwords Inggris: {len(stop_words_ing)}")


# Fungsi validasi stopword
def validate_stopwords():
    """Validasi bahwa file stopword berisi kata-kata yang diharapkan"""
    # Contoh kata stopword yang seharusnya ada
    expected_id_stopwords = {"yang", "dan", "di", "dari", "ke"}
    expected_ing_stopwords = {"the", "and", "is", "in", "to"}

    # Periksa stopword Indonesia
    missing_id = expected_id_stopwords - stop_words_id
    if missing_id:
        print(f"[WARNING] Stopword ID yang hilang: {missing_id}")

    # Periksa stopword Inggris
    missing_ing = expected_ing_stopwords - stop_words_ing
    if missing_ing:
        print(f"[WARNING] Stopword ING yang hilang: {missing_ing}")


# Panggil fungsi validasi
validate_stopwords()

# Load normalisasi dictionary
normalisasi_dict = {}
with open(os.path.join(TXT_DIR, "normalisasi_list.txt"), "r", encoding="utf-8") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            # Pastikan format konsisten (huruf kecil)
            k = k.strip().lower()
            v = v.strip().lower()
            normalisasi_dict[k] = v

# --- Load stemming_list.txt dengan handling yang lebih baik ---
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

    print(
        f"[STEMMING LIST] Loaded {len(stemming_dict)} pasangan dari stemming_list.txt"
    )
else:
    print("[STEMMING LIST] File tidak ditemukan, gunakan default Sastrawi.")

# Load game terms
game_terms = set()
with open(os.path.join(TXT_DIR, "game_term.txt"), "r", encoding="utf-8") as f:
    for line in f:
        term = line.strip().lower()
        if term:
            game_terms.add(term)
print(f"[GAME TERMS] Loaded {len(game_terms)} terms")

# Load kata tidak relevan
kata_tidak_relevan = set()
with open(os.path.join(TXT_DIR, "kata_tidak_relevan.txt"), "r", encoding="utf-8") as f:
    for line in f:
        word = line.strip().lower()
        if word:
            kata_tidak_relevan.add(word)
print(f"[KATA TIDAK RELEVAN] Loaded {len(kata_tidak_relevan)} words")

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
        print(f"[ERROR simpan cache bahasa]: {e}")


# --- Fungsi untuk mendeteksi pesan error ---
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
        print(f"[ERROR simpan cache]: {e}")


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
            print(
                f"[CACHE CLEANUP] Menghapus {len(keys_to_remove)} entri error dari cache"
            )
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

special_char_pattern = re.compile(r"[-–—…“»«]")
bracket_pattern = re.compile(r"\[.*?\]")
url_pattern = re.compile(r"http\S+")
digit_pattern = re.compile(r"\b\d+\b")
non_word_pattern = re.compile(r"[^\w\s@#]")
whitespace_pattern = re.compile(r"\s+")
repeated_word_pattern = re.compile(r"\b(\w{3,}?)(?:\1)\b")
sentence_split_pattern = re.compile(r"(?<=[.!?]) +|\n")
word_pattern = re.compile(r"\b\w+\b")

# --- Pattern untuk membersihkan token tidak diinginkan ---
# Pattern untuk angka dengan satuan (10k, 1k, 5v5, dll)
number_unit_pattern = re.compile(r"^\d+[a-z]*\d*[a-z]*$")
short_word_pattern = re.compile(r"^\w{1,2}$")  # Pattern untuk kata sangat pendek
mixed_alnum_pattern = re.compile(
    r"^[a-z]+\d+|\d+[a-z]+$"
)  # Pattern untuk kombinasi huruf dan angka


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


# --- Fungsi untuk normalisasi pengulangan huruf ---
def normalize_repeated_letters(word):
    """Mengurangi pengulangan huruf yang berlebihan menjadi maksimal 1 huruf"""
    if len(word) <= 2:
        return word

    # Gunakan regex untuk mengurangi pengulangan huruf menjadi 1
    # Contoh: "noooo" -> "no", "funnnnn" -> "fun"
    normalized = re.sub(r"(.)\1+", r"\1", word)
    return normalized


# --- Fungsi dengan mekanisme retry untuk terjemahan ---
def translate_with_retry(text, source="auto", target="id", max_len=5000):
    """Fungsi terjemahan dengan mekanisme retry untuk menangani connection lost"""
    if is_error_message(text):
        return text.lower()

    # Jeda acak antara permintaan untuk mengurangi beban server
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
                            print(f"[ERROR translate chunk]: {e}")
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
                    print(f"[ERROR translate chunk]: {e}")
                    hasil.append(buffer.strip())

            return " ".join(hasil)

        except Exception as e:
            error_msg = str(e).lower()
            print(f"[ATTEMPT {attempt + 1}/{MAX_RETRIES}] Error dalam terjemahan: {e}")

            # Deteksi apakah error terkait koneksi
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
                # Jika bukan error koneksi atau sudah mencapai batas retry
                raise e

            # Jika error koneksi, tunggu sebentar dan coba lagi
            print(f"Menunggu {delay} detik sebelum mencoba lagi...")
            time.sleep(delay)

            # Exponential backoff dengan jitter
            delay = min(MAX_DELAY, delay * 2 + random.uniform(0, 1))

    # Fallback ke teks asli jika semua percobaan gagal
    return text.lower()


def translate_long_text(text, source="auto", target="id", max_len=5000):
    """Wrapper untuk fungsi translate_with_retry"""
    return translate_with_retry(text, source, target, max_len)


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
        print(f"[ERROR bersihkan_terjemahan]: {e}")
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


# --- Fungsi untuk membersihkan token tidak diinginkan ---
def bersihkan_token(tokens):
    """Membersihkan token-token yang tidak diinginkan seperti angka dengan satuan, kata sangat pendek, dll."""
    hasil = []
    for token in tokens:
        # Normalisasi pengulangan huruf terlebih dahulu
        normalized_token = normalize_repeated_letters(token)

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


# --- Fungsi Stopword Removal dengan Prioritas yang Diperbarui ---
def hapus_stopword(words, debug=False):
    # Bersihkan token tidak diinginkan terlebih dahulu
    words = bersihkan_token(words)

    if debug:
        print(f"[DEBUG STOPWORD] Kata sebelum filter: {words}")

    result = []
    for w in words:
        # Pengecualian: Jika kata adalah game term, selalu pertahankan
        if w in game_terms:
            if debug:
                print(f"[DEBUG STOPWORD] Menyimpan game term: '{w}'")
            result.append(w)
            continue

        # Prioritas 1: Cek kata tidak relevan terlebih dahulu (jika bukan stopword Indonesia)
        if w not in stop_words_id and w in kata_tidak_relevan:
            if debug:
                print(f"[DEBUG STOPWORD] Menghapus kata tidak relevan: '{w}'")
            continue

        # Prioritas 2: Cek stopword Inggris
        if w in stop_words_ing:
            if debug:
                print(f"[DEBUG STOPWORD] Menghapus stopword Inggris: '{w}'")
            continue

        # Prioritas 3: Cek stopword Indonesia
        if w in stop_words_id:
            if debug:
                print(f"[DEBUG STOPWORD] Menghapus stopword Indonesia: '{w}'")
            continue

        # Jika bukan stopword atau kata tidak relevan, simpan
        if debug:
            print(f"[DEBUG STOPWORD] Menyimpan kata: '{w}'")
        result.append(w)

    if debug:
        print(f"[DEBUG STOPWORD] Kata setelah filter: {result}")

    return result


# --- Fungsi Stemming yang Diperbarui ---
def stemming_teks(words, debug=False):
    hasil = []
    stem_methods = []  # Untuk melacak metode yang digunakan (opsional, untuk debugging)

    for w in words:
        wl = w.lower()
        wl_normalized = normalize_repeated_letters(wl)
        method_used = "original"  # Default

        # 1. PRIORITAS PERTAMA: Cek kamus custom (stemming_list.txt)
        if wl in stemming_dict:
            mapped = stemming_dict[wl]
            hasil.extend(mapped.split())
            method_used = "custom_dict"
        elif wl_normalized in stemming_dict:
            mapped = stemming_dict[wl_normalized]
            hasil.extend(mapped.split())
            method_used = "custom_dict_normalized"
        else:
            # 2. PRIORITAS KEDUA: Gunakan Sastrawi
            stemmed_sastrawi = cached_stemmer_stem(wl_normalized)

            if stemmed_sastrawi != wl_normalized:
                hasil.append(stemmed_sastrawi)
                method_used = "sastrawi"
            else:
                # Jika semua metode gagal, gunakan kata asli yang sudah dinormalisasi
                hasil.append(wl_normalized)
                method_used = "normalized"

        stem_methods.append(method_used)

    # Log metode yang digunakan (opsional, untuk debugging)
    if debug and stem_methods:
        method_counts = Counter(stem_methods)
        print(f"[STEMMING METHODS] Digunakan: {dict(method_counts)}")

    return hasil


def deteksi_bukan_indonesia(text: str) -> bool:
    try:
        return cached_detect_language(text) != "id"
    except Exception:
        return False


WHITELIST_FILE = os.path.join(TXT_DIR, "kata_ambigu.txt")
kata_id_pasti = set()

if os.path.exists(WHITELIST_FILE):
    try:
        with open(WHITELIST_FILE, "r", encoding="utf-8") as f:
            kata_id_pasti = {line.strip().lower() for line in f if line.strip()}
        print(f"[WHITELIST] Loaded {len(kata_id_pasti)} kata dari kata_ambigu.txt")
    except Exception as e:
        print(f"[ERROR baca whitelist]: {e}")
else:
    print("[WHITELIST] File kata_ambigu.txt tidak ditemukan, whitelist kosong.")


# --- Fungsi untuk mendeteksi kata asing dalam teks ---
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
        print(f"[ERROR deteksi kata asing]: {e}")

    return False


# --- Fungsi preprocessing standar yang digunakan oleh kedua fase ---
def proses_preprocessing_standar(teks, debug=False):
    """Fungsi preprocessing standar yang digunakan oleh kedua fase dengan stopword gabungan"""
    if pd.isna(teks) or not isinstance(teks, str) or not teks.strip():
        return ["", "", [], [], [], [], ""]

    # Bersihkan teks
    clean = bersihkan_terjemahan(teks)
    folded = clean.lower()

    # Tokenisasi
    try:
        token = word_tokenize(folded)
    except Exception:
        token = word_pattern.findall(folded)

    # Bersihkan token tidak diinginkan
    token = bersihkan_token(token)

    if debug:
        print(f"[DEBUG PREPRO] Setelah tokenisasi: {token}")

    # Proses NLP dengan stopword gabungan
    stop = hapus_stopword(token, debug) if token else []

    if debug:
        print(f"[DEBUG PREPRO] Setelah stopword removal: {stop}")

    norm = normalisasi_teks(stop) if stop else []

    if debug:
        print(f"[DEBUG PREPRO] Setelah normalisasi: {norm}")

    stem = stemming_teks(norm, debug) if norm else []

    if debug:
        print(f"[DEBUG PREPRO] Setelah stemming: {stem}")

    # Gabungkan hasil
    hasil = " ".join(stem) if stem else folded

    return [clean, folded, token, stop, norm, stem, hasil]


# --- Proses baris dengan preprocessing standar ---
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
        print(f"[ERROR proses standar]: {e}")
        clean = bersihkan_terjemahan(terjemahan) if isinstance(terjemahan, str) else ""
        folded = clean.lower() if clean else ""
        hasil = folded if folded else (clean if clean else str(terjemahan))
        return [clean, folded, [], [], [], [], hasil]


# --- Proses batch untuk preprocessing standar ---
def proses_batch_standar(terjemahan_list, debug=False):
    """Proses batch data untuk preprocessing standar"""
    hasil = []
    for terjemahan in terjemahan_list:
        hasil.append(proses_baris_standar(terjemahan, debug))
    return hasil


# --- Fungsi untuk terjemahan ulang dengan retry ---
def proses_terjemahan_ulang(teks_asal, hasil_sebelumnya, data_preprocessing_sebelumnya):
    """Melakukan terjemahan ulang hanya jika diperlukan"""
    # Periksa apakah teks mengandung kata asing
    if not contains_foreign_words(hasil_sebelumnya):
        return data_preprocessing_sebelumnya

    try:
        # Terjemahkan ulang teks asal dengan mekanisme retry
        terjemahan_baru = translate_with_retry(teks_asal)

        # Jika terjemahan baru error atau kosong, kembalikan hasil sebelumnya
        if not terjemahan_baru or is_error_message(terjemahan_baru):
            return data_preprocessing_sebelumnya

        # Gunakan fungsi preprocessing standar yang sama untuk hasil terjemahan baru
        return proses_preprocessing_standar(terjemahan_baru)

    except Exception as e:
        print(f"[ERROR terjemahan ulang setelah {MAX_RETRIES} percobaan]: {e}")
        return data_preprocessing_sebelumnya


def save_csv(df: pd.DataFrame, file_path: str = PREPRO_CSV_PATH):
    """Menyimpan DataFrame ke CSV dengan format konsisten."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        df_for_file = df.copy(deep=True)

        # Konversi kolom list menjadi string yang aman untuk CSV
        list_cols = ["Tokenisasi", "Stopword", "Normalisasi", "Stemming"]
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
                        # Escape kutipan dan format untuk CSV
                        escaped_value = value.replace('"', '""')
                        row_data.append('"' + escaped_value + '"')
                    else:
                        row_data.append('"' + str(value) + '"')
                f.write(",".join(row_data) + "\r\n")

        print(f"[SAVE CSV] File berhasil disimpan di {file_path}")
        return True
    except Exception as e:
        print(f"[ERROR save_csv]: {e}")
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
        print("Fase 1: Preprocessing standar...")
        with ProcessPoolExecutor(max_workers=4) as executor:
            chunks = [df[i : i + CHUNK_SIZE] for i in range(0, total_rows, CHUNK_SIZE)]
            futures = []

            for i, chunk in enumerate(chunks):
                futures.append(
                    executor.submit(proses_batch_standar, chunk["Terjemahan"].tolist())
                )

            processed_chunks = []
            for i, future in enumerate(as_completed(futures)):
                print(f"[CHUNK {i + 1}] Memproses {len(chunks[i])} baris...")
                hasil_list = future.result()
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
        print("Fase 2: Terjemahan ulang...")
        baris_diperbaiki = 0
        for idx in range(len(result_df)):
            if idx % 100 == 0:
                print(f"Memeriksa baris {idx + 1} dari {len(result_df)}")

            teks_asal = result_df.at[idx, "Terjemahan"]
            hasil_sebelumnya = result_df.at[idx, "Hasil"]

            # Siapkan data preprocessing sebelumnya untuk fallback
            data_preprocessing_sebelumnya = [
                result_df.at[idx, "Clean Data"],
                result_df.at[idx, "Case Folding"],
                result_df.at[idx, "Tokenisasi"],
                result_df.at[idx, "Stopword"],
                result_df.at[idx, "Normalisasi"],
                result_df.at[idx, "Stemming"],
                hasil_sebelumnya,
            ]

            # Terjemahkan ulang jika diperlukan
            hasil_baru = proses_terjemahan_ulang(
                teks_asal, hasil_sebelumnya, data_preprocessing_sebelumnya
            )

            # Jika hasil baru berbeda dari sebelumnya, update semua kolom
            if hasil_baru[6] != hasil_sebelumnya:
                # Update semua kolom dengan hasil baru
                result_df.at[idx, "Clean Data"] = hasil_baru[0]
                result_df.at[idx, "Case Folding"] = hasil_baru[1]
                result_df.at[idx, "Tokenisasi"] = hasil_baru[2]
                result_df.at[idx, "Stopword"] = hasil_baru[3]
                result_df.at[idx, "Normalisasi"] = hasil_baru[4]
                result_df.at[idx, "Stemming"] = hasil_baru[5]
                result_df.at[idx, "Hasil"] = hasil_baru[6]
                baris_diperbaiki += 1
            # Jika tidak ada perubahan, pastikan semua kolom terisi dengan data sebelumnya
            else:
                result_df.at[idx, "Clean Data"] = data_preprocessing_sebelumnya[0]
                result_df.at[idx, "Case Folding"] = data_preprocessing_sebelumnya[1]
                result_df.at[idx, "Tokenisasi"] = data_preprocessing_sebelumnya[2]
                result_df.at[idx, "Stopword"] = data_preprocessing_sebelumnya[3]
                result_df.at[idx, "Normalisasi"] = data_preprocessing_sebelumnya[4]
                result_df.at[idx, "Stemming"] = data_preprocessing_sebelumnya[5]
                result_df.at[idx, "Hasil"] = data_preprocessing_sebelumnya[6]

        print(f"Terjemahan ulang selesai. {baris_diperbaiki} baris diperbaiki.")

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
        print(f"[DEBUG] Memproses teks: {teks}")

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
        return jsonify({"error": str(e)}), 500


@prepro_bp.route("/")
def index():
    return render_template("preprosessing.html", page_name="prepro"), 200

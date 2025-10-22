import json
import os
import re
import pandas as pd
from flask import Blueprint, request, jsonify, render_template, send_file
from nltk.tokenize import word_tokenize
from markupsafe import escape
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from deep_translator import GoogleTranslator
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from functools import lru_cache
import random
import logging

# IMPORT MODUL PREPROCESSING BARU
from preprocessing_module import preprocessor

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

# Deteksi environment Flask
IS_MAIN_PROCESS = os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not os.environ.get(
    "WERKZEUG_RUN_MAIN"
)

# Konfigurasi retry dan jeda
MAX_RETRIES = 3
INITIAL_DELAY = 1
MAX_DELAY = 10
REQUEST_DELAY = 0.5

# Path cache files
CACHE_FOLDER = os.path.join(BASE_DIR, "uploads", "preproses")
CACHE_FILE = os.path.join(CACHE_FOLDER, "cache_translate.json")
LANGUAGE_CACHE_FILE = os.path.join(CACHE_FOLDER, "cache_language.json")
TRANSLATION_CACHE_FILE = os.path.join(
    BASE_DIR, "uploads", "terjemahan", "translation_cache.json"
)

CHUNK_SIZE = 1000
MAX_FILE_SIZE = 2 * 1024 * 1024


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


# --- Optimasi: Gunakan LRU cache untuk fungsi yang sering dipanggil ---
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
            kalimat = preprocessor.sentence_split_pattern.split(text)

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


def hapus_kata_ulang(word):
    return preprocessor.repeated_word_pattern.sub(r"\1", word)


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
                or token in preprocessor.game_terms  # Game terms dipertahankan
                or token in preprocessor.kata_id_pasti  # Kata whitelist dipertahankan
                or token
                in preprocessor.normalisasi_dict  # Kata yang sudah dinormalisasi
                or token in preprocessor.stemming_dict  # Kata yang sudah distem
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


# --- FUNGSI PREPROCESSING STANDAR DENGAN MODUL ---
def proses_preprocessing_standar(teks, debug=False):
    """Wrapper untuk modul preprocessing"""
    return preprocessor.proses_preprocessing_standar(teks, debug)


def proses_baris_standar(terjemahan, debug=False):
    try:
        if (
            pd.isna(terjemahan)
            or not isinstance(terjemahan, str)
            or not terjemahan.strip()
        ):
            clean_minimal = (
                preprocessor.whitespace_pattern.sub(" ", str(terjemahan)).strip()
                if isinstance(terjemahan, str)
                else ""
            )
            folded_minimal = clean_minimal.lower()
            return [clean_minimal, folded_minimal, [], [], [], [], folded_minimal]

        return proses_preprocessing_standar(terjemahan, debug)

    except Exception as e:
        logging.error(f"Error proses standar: {e}")
        clean = (
            preprocessor.bersihkan_terjemahan(terjemahan)
            if isinstance(terjemahan, str)
            else ""
        )
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

        # GUNAKAN MODUL untuk preprocessing
        return preprocessor.proses_preprocessing_standar(terjemahan_baru, debug=False)

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
        preprocessor.load_normalization_dict()

        # Test normalisasi
        tokens = word_tokenize(test_text.lower())
        normalized = preprocessor.normalisasi_teks(tokens, debug=True)

        # Analisis detail
        analysis = []
        for token in tokens:
            in_dict = token in preprocessor.normalisasi_dict
            mapped_to = preprocessor.normalisasi_dict.get(token, "TIDAK_ADA")
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
                "dictionary_sample": dict(
                    list(preprocessor.normalisasi_dict.items())[:10]
                ),
                "test_keys_presence": {
                    "bgtt": "bgtt" in preprocessor.normalisasi_dict,
                    "bs": "bs" in preprocessor.normalisasi_dict,
                    "td": "td" in preprocessor.normalisasi_dict,
                    "klo": "klo" in preprocessor.normalisasi_dict,
                    "yg": "yg" in preprocessor.normalisasi_dict,
                    "dgn": "dgn" in preprocessor.normalisasi_dict,
                    "gue": "gue" in preprocessor.normalisasi_dict,
                    "lo": "lo" in preprocessor.normalisasi_dict,
                },
            }
        )

    except Exception as e:
        logging.error(f"Error test_normalization: {e}")
        return jsonify({"error": str(e)}), 500


@prepro_bp.route("/validate_dictionary", methods=["GET"])
def validate_dictionary():
    """Validasi isi kamus normalisasi"""
    sample_size = min(20, len(preprocessor.normalisasi_dict))
    sample_items = dict(list(preprocessor.normalisasi_dict.items())[:sample_size])

    # Cek contoh kata yang seharusnya dinormalisasi
    test_cases = ["gue", "lo", "bgtt", "bs", "td", "klo", "yg", "dgn", "ntar", "ga"]
    test_results = {}

    for test_word in test_cases:
        test_results[test_word] = {
            "in_dict": test_word in preprocessor.normalisasi_dict,
            "mapped_to": preprocessor.normalisasi_dict.get(test_word, "TIDAK_ADA"),
        }

    return jsonify(
        {
            "dictionary_size": len(preprocessor.normalisasi_dict),
            "sample_items": sample_items,
            "test_cases": test_results,
            "common_missing": [
                k for k in test_cases if k not in preprocessor.normalisasi_dict
            ],
        }
    )


# --- Routes yang diperbaiki ---
@prepro_bp.route("/preproses", methods=["POST"])
def preproses():
    try:
        # TIDAK PERLU inisialisasi manual - modul sudah auto-initialize

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

        # Fase 1: Preprocessing standar dengan modul
        logging.info("Fase 1: Preprocessing standar (dengan modul)...")
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

        # Lakukan preprocessing dengan debug mode menggunakan MODUL
        hasil = preprocessor.proses_preprocessing_standar(teks, debug=True)

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

        # Cek normalisasi untuk setiap token menggunakan MODUL
        normalization_results = []
        for token in tokens:
            token_lower = token.lower()
            is_in_dict = token_lower in preprocessor.normalisasi_dict
            normalized_value = preprocessor.normalisasi_dict.get(
                token_lower, "TIDAK_ADA"
            )
            normalization_results.append(
                {
                    "token": token,
                    "lowercase": token_lower,
                    "in_normalization_dict": is_in_dict,
                    "normalized_to": normalized_value,
                }
            )

        # Proses lengkap untuk perbandingan menggunakan MODUL
        hasil_lengkap = preprocessor.proses_preprocessing_standar(teks, debug=True)

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
                "normalization_dict_sample": dict(
                    list(preprocessor.normalisasi_dict.items())[:10]
                ),
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

        # Re-initialize data MODUL
        preprocessor.initialize_preprocessing_data()

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
        hasil = preprocessor.proses_preprocessing_standar(test_text, debug=True)

        return jsonify(
            {
                "status": "success",
                "test_text": test_text,
                "result": hasil[6],  # Hasil akhir
                "stopwords_loaded": {
                    "indonesia_count": len(preprocessor.stop_words_id),
                    "english_count": len(preprocessor.stop_words_ing),
                    "sample_id": list(preprocessor.stop_words_id)[:10],
                    "sample_ing": list(preprocessor.stop_words_ing)[:10],
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
        hasil = preprocessor.proses_preprocessing_standar(teks, debug=True)

        # Analisis tambahan
        tokens = word_tokenize(teks.lower())
        normalization_analysis = []
        foreign_analysis = []

        for token in tokens:
            # Analisis normalisasi
            if token in preprocessor.normalisasi_dict:
                normalization_analysis.append(
                    {
                        "token": token,
                        "in_dict": True,
                        "normalized_to": preprocessor.normalisasi_dict[token],
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
                and token not in preprocessor.game_terms
                and token not in preprocessor.kata_id_pasti
                and token not in preprocessor.normalisasi_dict
                and token not in preprocessor.stemming_dict
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
                    "normalization_dict_size": len(preprocessor.normalisasi_dict),
                    "game_terms_size": len(preprocessor.game_terms),
                    "whitelist_size": len(preprocessor.kata_id_pasti),
                    "sample_normalization": dict(
                        list(preprocessor.normalisasi_dict.items())[:10]
                    ),
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

        in_normalization = word in preprocessor.normalisasi_dict
        in_stemming = word in preprocessor.stemming_dict
        in_game_terms = word in preprocessor.game_terms
        in_whitelist = word in preprocessor.kata_id_pasti

        return jsonify(
            {
                "word": word,
                "in_normalization_dict": in_normalization,
                "normalization_value": preprocessor.normalisasi_dict.get(word)
                if in_normalization
                else None,
                "in_stemming_dict": in_stemming,
                "stemming_value": preprocessor.stemming_dict.get(word)
                if in_stemming
                else None,
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

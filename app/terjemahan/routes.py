import os
import csv
import re
import time
import json
import logging
from flask import Blueprint, request, jsonify, render_template, send_file
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

terjemahan_bp = Blueprint(
    "terjemahan",
    __name__,
    url_prefix="/terjemahan",
    template_folder="templates",
    static_folder="static",
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "terjemahan")
TRANSLATED_PATH = os.path.join(UPLOAD_FOLDER, "terjemahan_steam_reviews.csv")
CACHE_FILE = os.path.join(UPLOAD_FOLDER, "translation_cache.json")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

translated_data = []
ignored_words = set()
translation_cache = {}
cache_lock = Lock()

# =====================
# Cache Functions
# =====================
def load_cache():
    global translation_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                translation_cache = json.load(f)
            logging.info(f"Cache loaded: {len(translation_cache)} entries")
        except Exception as e:
            logging.error(f"Failed to load cache: {e}")
            translation_cache = {}

def save_cache():
    with cache_lock:
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(translation_cache, f, ensure_ascii=False, indent=2)
            logging.info(f"Cache saved: {len(translation_cache)} entries")
        except Exception as e:
            logging.error(f"Failed to save cache: {e}")

# =====================
# Utilities
# =====================
def load_ignored_words():
    global ignored_words
    file_path = os.path.join(os.path.dirname(__file__), "abai_singkat.txt")
    ignored_words.clear()
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            ignored_words = {word.strip().lower() for word in f if word.strip()}
    logging.info(f"Loaded {len(ignored_words)} ignored words.")

def split_text_into_chunks(text, max_chars=5000):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > max_chars:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def fix_hyphen_and_detect_reduplication(text):
    def is_kata_ulang(a, b):
        a, b = a.lower(), b.lower()
        if a == b:
            return True
        if len(a) == len(b):
            beda = sum(1 for x, y in zip(a, b) if x != y)
            if beda <= 2:
                return True
        if a.startswith(b) or b.startswith(a):
            return True
        if a.endswith(b) or b.endswith(a):
            return True
        return False

    def koreksi(match):
        k1, k2 = match.group(1), match.group(2)
        if is_kata_ulang(k1, k2):
            return f"{k1} - {k2}"
        return f"{k1} {k2}"

    return re.sub(r"\b(\w+)\s*-\s*(\w+)\b", koreksi, text)

def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"\[/?[a-zA-Z0-9]+\]", "", text)
    text = text.strip()
    if re.fullmatch(r"[\W_]+", text):
        return ""
    return text

def translate_chunk(chunk, target_language="id", retries=3, delay=1.0) -> str:
    original_chunk = chunk
    chunk = clean_text(chunk)
    if not chunk:
        logging.warning(f"Chunk kosong setelah clean_text: '{original_chunk[:80]}...'")
        return "[Gagal diterjemahkan]"
    
    for attempt in range(1, retries + 1):
        try:
            translated = GoogleTranslator(source="auto", target=target_language).translate(chunk)
            if not translated or translated.strip() == "":
                logging.warning(f"Terjemahan kosong untuk chunk: '{chunk[:80]}...'")
                continue

            # 🔴 Tambahan filter: jangan terima hasil error server
            if any(err in translated for err in ["Error 500", "Server Error", "Please try again later"]):
                logging.error(f"Server error diterima sebagai teks: '{translated[:80]}...'")
                if attempt < retries:
                    time.sleep(delay * attempt)
                    continue
                return "[Gagal diterjemahkan]"

            return fix_hyphen_and_detect_reduplication(translated)

        except Exception as e:
            logging.warning(
                f"Percobaan {attempt} gagal terjemahkan chunk (potongan: {chunk[:80]}...): {e}"
            )
            if attempt < retries:
                time.sleep(delay * attempt)

    return "[Gagal diterjemahkan]"

def translate_large_text(text, target_language="id") -> str:
    if not text or not isinstance(text, str) or text.strip() == "":
        return ""
    chunks = split_text_into_chunks(text, 5000)
    translated_chunks = [translate_chunk(c, target_language) for c in chunks]
    return " ".join(translated_chunks).strip()

@terjemahan_bp.route("/translate", methods=["POST"])
def translate_file():
    try:
        global translated_data
        file = request.files.get("file")
        if not file or not file.filename or not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Jenis file tidak valid. Harus .csv"}), 400

        content = file.read().decode("utf-8").splitlines()
        reader = csv.DictReader(content)
        if reader.fieldnames is None:
            return jsonify({"error": "File CSV tidak memiliki header."}), 400

        header = [col.strip('"').strip() for col in reader.fieldnames]
        required_columns = {"SteamID", "Ulasan", "Status"}
        if not required_columns.issubset(header):
            missing = required_columns - set(header)
            return jsonify({"error": f"Kolom wajib hilang: {', '.join(missing)}"}), 400

        load_ignored_words()
        load_cache()
        translated_data = []
        rows = list(reader)

        # Counter untuk tracking progres
        processed_count = 0
        failed_count = 0
        progress_lock = Lock()

        def translate_row(row):
            nonlocal processed_count, failed_count
            steamid = row.get("SteamID", "")
            ulasan = row.get("Ulasan") or ""
            status = row.get("Status", "")
            if not isinstance(ulasan, str):
                ulasan = ""
            
            terjemahan = ""
            if ulasan.strip() != "":
                with cache_lock:
                    if ulasan in translation_cache:
                        terjemahan = translation_cache[ulasan]
                    else:
                        terjemahan = translate_large_text(ulasan, "id")
                        translation_cache[ulasan] = terjemahan

                # Cek jika terjemahan gagal
                if terjemahan == "[Gagal diterjemahkan]":
                    with progress_lock:
                        failed_count += 1
                        logging.error(f"Gagal menerjemahkan SteamID {steamid}")

            # Update counter dan log setiap 100 baris
            with progress_lock:
                processed_count += 1
                if processed_count % 100 == 0:
                    logging.info(f"Berhasil memproses {processed_count} baris")

            return {
                "SteamID": steamid,
                "Ulasan": ulasan,
                "Terjemahan": terjemahan,
                "Status": status,
            }

        with ThreadPoolExecutor(max_workers=2) as executor:
            translated_data = list(executor.map(translate_row, rows))

        # Log hasil akhir
        logging.info(f"Total baris diproses: {processed_count}")
        logging.info(f"Total gagal diterjemahkan: {failed_count}")

        save_cache()

        os.makedirs(os.path.dirname(TRANSLATED_PATH), exist_ok=True)
        with open(TRANSLATED_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["SteamID", "Ulasan", "Terjemahan", "Status"])
            for row in translated_data:
                if not row["Ulasan"].strip():
                    continue
                writer.writerow([
                    row["SteamID"],
                    row["Ulasan"].replace("\n", " ").replace("\r", " "),
                    row["Terjemahan"],
                    row["Status"],
                ])

        return jsonify(
            {"message": f"Berhasil diterjemahkan! Diproses: {processed_count}, Gagal: {failed_count}", "data": translated_data}
        ), 200

    except UnicodeDecodeError:
        return jsonify({"error": "Encoding file harus UTF-8."}), 400
    except Exception as e:
        logging.exception("Kesalahan tak terduga saat proses terjemahan")
        return jsonify({"error": f"Kesalahan server: {str(e)}"}), 500


@terjemahan_bp.route("/cleardata", methods=["POST"])
def clear_data():
    global translated_data
    translated_data = []
    return jsonify({"message": "Data cleared successfully"})

@terjemahan_bp.route("/savedata", methods=["GET"])
def savedata():
    global translated_data
    if not translated_data:
        return jsonify({"status": "error", "message": "Tidak ada data untuk disimpan."})
    os.makedirs(os.path.dirname(TRANSLATED_PATH), exist_ok=True)
    try:
        with open(TRANSLATED_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["SteamID", "Ulasan", "Terjemahan", "Status"])
            for row in translated_data:
                if not row["Ulasan"].strip():
                    continue
                writer.writerow([
                    row["SteamID"],
                    row["Ulasan"].replace("\n", " ").replace("\r", " "),
                    row["Terjemahan"],
                    row["Status"],
                ])
        return send_file(
            TRANSLATED_PATH,
            mimetype="text/csv",
            as_attachment=True,
            download_name="terjemahan_steam_reviews.csv",
        )
    except Exception as e:
        return jsonify({"status": "error", "message": f"Gagal menyimpan file: {e}"}), 500

@terjemahan_bp.route("/download")
def download_latest():
    if os.path.exists(TRANSLATED_PATH):
        return send_file(TRANSLATED_PATH, as_attachment=True)
    return jsonify({"error": "File hasil terjemahan tidak ditemukan."}), 404

@terjemahan_bp.route("/")
def index():
    return render_template("terjemahan.html", page_name="terjemahan")

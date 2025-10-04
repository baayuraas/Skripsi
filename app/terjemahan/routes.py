import os
import csv
import re
import time
import json
import logging
from flask import Blueprint, request, jsonify, render_template, send_file
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

terjemahan_bp = Blueprint(
    "terjemahan",
    __name__,
    url_prefix="/terjemahan",
    template_folder="templates",
    static_folder="static",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "translation.log")),
        logging.StreamHandler(),
    ],
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
# Configuration - OPTIMIZED FOR SPEED
# =====================
TRANSLATION_CONFIG = {
    "max_workers": 2,  # Balanced between speed and rate limiting
    "request_delay": 0.3,  # Delay between requests
    "chunk_size": 4500,  # Google's limit is 5000
    "batch_size": 15,  # Process in batches
    "max_retries": 2,  # Reduced retries for speed
    "retry_delay": 1.0,
}


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
# Utilities - OPTIMIZED
# =====================
def load_ignored_words():
    global ignored_words
    file_path = os.path.join(os.path.dirname(__file__), "abai_singkat.txt")
    ignored_words.clear()
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            ignored_words = {word.strip().lower() for word in f if word.strip()}
    logging.info(f"Loaded {len(ignored_words)} ignored words.")


def split_text_into_chunks(text, max_chars=4500):
    """Efficient chunking with sentence boundaries"""
    if len(text) <= max_chars:
        return [text]

    # Split by sentences when possible
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # If single sentence is too long, split by length
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i : i + max_chars])
                current_chunk = ""
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def clean_text(text: str) -> str:
    """Fast text cleaning"""
    if not text or not isinstance(text, str):
        return ""

    # Quick cleanup
    text = re.sub(r"\[/?[^\]]+\]", "", text)  # Remove Steam tags
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    text = text.strip()

    if not text or re.fullmatch(r"[\W_]+", text):
        return ""
    return text


def translate_chunk(chunk, target_language="id"):
    """Fast translation with minimal overhead - FIXED VERSION"""
    chunk = clean_text(chunk)
    if not chunk:
        return ""

    for attempt in range(TRANSLATION_CONFIG["max_retries"]):
        try:
            translated = GoogleTranslator(
                source="auto", target=target_language
            ).translate(chunk)

            if not translated or not translated.strip():
                continue

            # Quick error check
            if any(
                err in translated
                for err in ["Error 500", "Server Error", "Please try again"]
            ):
                continue

            return translated

        except Exception as e:  # FIXED: & changed to as
            logging.warning(
                f"Attempt {attempt + 1} failed for chunk (length: {len(chunk)}): {e}"
            )
            if (
                attempt < TRANSLATION_CONFIG["max_retries"] - 1
            ):  # FIXED: -4 changed to -1
                time.sleep(TRANSLATION_CONFIG["retry_delay"] * (attempt + 1))
            continue

    return ""


def translate_text(text, target_language="id"):
    """Main translation function with cache support"""
    if not text or not isinstance(text, str) or text.strip() == "":
        return ""

    # Check cache first
    cache_key = text.strip()
    with cache_lock:
        if cache_key in translation_cache:
            return translation_cache[cache_key]

    chunks = split_text_into_chunks(text, TRANSLATION_CONFIG["chunk_size"])

    # Single chunk - direct translation
    if len(chunks) == 1:
        translated = translate_chunk(chunks[0], target_language)
        with cache_lock:
            translation_cache[cache_key] = translated
        return translated

    # Multiple chunks
    translated_chunks = []
    for chunk in chunks:
        translated = translate_chunk(chunk, target_language)
        if translated:
            translated_chunks.append(translated)
        time.sleep(TRANSLATION_CONFIG["request_delay"])

    result = " ".join(translated_chunks).strip()
    with cache_lock:
        translation_cache[cache_key] = result

    return result


def process_batch(batch_rows, batch_num, total_batches, progress_data):
    """Process a batch of rows with progress tracking"""
    batch_results = []

    for row in batch_rows:
        steamid = row.get("SteamID", "")
        ulasan = row.get("Ulasan") or ""
        status = row.get("Status", "")

        if not isinstance(ulasan, str):
            ulasan = ""

        terjemahan = ""
        if ulasan.strip():
            # Check cache first
            with cache_lock:
                if ulasan in translation_cache:
                    terjemahan = translation_cache[ulasan]
                    progress_data["cached_count"] += 1
                else:
                    terjemahan = translate_text(ulasan, "id")
                    translation_cache[ulasan] = terjemahan
                    if terjemahan:
                        progress_data["translated_count"] += 1
                    else:
                        progress_data["failed_count"] += 1

        batch_results.append(
            {
                "SteamID": steamid,
                "Ulasan": ulasan,
                "Terjemahan": terjemahan,
                "Status": status,
            }
        )

        progress_data["processed_count"] += 1

        # Progress logging every 10 rows
        if progress_data["processed_count"] % 10 == 0:
            logging.info(
                f"Batch {batch_num}/{total_batches}: "
                f"Processed {progress_data['processed_count']}/"
                f"{progress_data['total_rows']} "
                f"({progress_data['processed_count'] / progress_data['total_rows'] * 100:.1f}%)"
            )

    return batch_results


@terjemahan_bp.route("/translate", methods=["POST"])
def translate_file():
    start_time = time.time()
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

        rows = list(reader)
        total_rows = len(rows)
        translated_data = []

        # Progress tracking
        progress_data = {
            "processed_count": 0,
            "translated_count": 0,
            "cached_count": 0,
            "failed_count": 0,
            "total_rows": total_rows,
        }

        logging.info(
            f"Memulai terjemahan {total_rows} baris dengan {TRANSLATION_CONFIG['max_workers']} workers"
        )

        # Split into batches
        batch_size = TRANSLATION_CONFIG["batch_size"]
        batches = [rows[i : i + batch_size] for i in range(0, total_rows, batch_size)]
        total_batches = len(batches)

        # Process batches in parallel
        with ThreadPoolExecutor(
            max_workers=TRANSLATION_CONFIG["max_workers"]
        ) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(
                    process_batch, batch, i + 1, total_batches, progress_data
                ): i
                for i, batch in enumerate(batches)
            }

            # Collect results as they complete
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    translated_data.extend(batch_results)

                    # Save cache every 5 batches to prevent data loss
                    if len(translated_data) % (batch_size * 5) == 0:
                        save_cache()

                except Exception as e:
                    logging.error(f"Error processing batch: {e}")
                    continue

        # Final save
        save_cache()

        # Save to CSV file
        os.makedirs(os.path.dirname(TRANSLATED_PATH), exist_ok=True)
        with open(TRANSLATED_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["SteamID", "Ulasan", "Terjemahan", "Status"])
            for row in translated_data:
                if not row["Ulasan"].strip():
                    continue
                writer.writerow(
                    [
                        row["SteamID"],
                        row["Ulasan"].replace("\n", " ").replace("\r", " "),
                        row["Terjemahan"],
                        row["Status"],
                    ]
                )

        # Calculate statistics
        success_count = (
            progress_data["translated_count"] + progress_data["cached_count"]
        )
        total_time = time.time() - start_time

        logging.info(
            f"Terjemahan selesai dalam {total_time:.1f} detik: "
            f"{success_count} berhasil, {progress_data['failed_count']} gagal, "
            f"{progress_data['cached_count']} dari cache"
        )

        return jsonify(
            {
                "message": f"Berhasil diterjemahkan! "
                f"Waktu: {total_time:.1f} detik, "
                f"Total: {progress_data['processed_count']}, "
                f"Berhasil: {success_count}, "
                f"Gagal: {progress_data['failed_count']}, "
                f"Cache: {progress_data['cached_count']}",
                "data": translated_data,
                "stats": {
                    "total_time": round(total_time, 1),
                    "total_rows": progress_data["processed_count"],
                    "success_count": success_count,
                    "failed_count": progress_data["failed_count"],
                    "cached_count": progress_data["cached_count"],
                },
            }
        ), 200

    except UnicodeDecodeError:
        return jsonify({"error": "Encoding file harus UTF-8."}), 400
    except Exception as e:
        logging.exception("Kesalahan tak terduga saat proses terjemahan")
        return jsonify({"error": f"Kesalahan server: {str(e)}"}), 500


# ... (other routes remain the same as previous version)
@terjemahan_bp.route("/get_saved_data", methods=["GET"])
def get_saved_data():
    if not os.path.exists(TRANSLATED_PATH):
        return jsonify({"error": "File terjemahan tidak ditemukan."}), 404

    try:
        data = []
        with open(TRANSLATED_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(
                    {
                        "SteamID": row.get("SteamID", ""),
                        "Ulasan": row.get("Ulasan", ""),
                        "Terjemahan": row.get("Terjemahan", ""),
                        "Status": row.get("Status", ""),
                    }
                )

        return jsonify({"data": data}), 200
    except Exception as e:
        logging.error(f"Gagal membaca file terjemahan: {e}")
        return jsonify({"error": "Gagal membaca file terjemahan."}), 500


@terjemahan_bp.route("/cleardata", methods=["POST"])
def clear_data():
    global translated_data, translation_cache
    translated_data = []
    translation_cache = {}

    # Clear cache file
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

    # Clear translated file
    if os.path.exists(TRANSLATED_PATH):
        os.remove(TRANSLATED_PATH)

    return jsonify({"message": "Data dan cache berhasil dihapus"})


@terjemahan_bp.route("/savedata", methods=["GET"])
def savedata():
    global translated_data
    if not translated_data and os.path.exists(TRANSLATED_PATH):
        try:
            with open(TRANSLATED_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                translated_data = [
                    {
                        "SteamID": row.get("SteamID", ""),
                        "Ulasan": row.get("Ulasan", ""),
                        "Terjemahan": row.get("Terjemahan", ""),
                        "Status": row.get("Status", ""),
                    }
                    for row in reader
                ]
        except Exception as e:
            return jsonify(
                {"status": "error", "message": f"Gagal membaca file: {e}"}
            ), 500

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
                writer.writerow(
                    [
                        row["SteamID"],
                        row["Ulasan"].replace("\n", " ").replace("\r", " "),
                        row["Terjemahan"],
                        row["Status"],
                    ]
                )
        return send_file(
            TRANSLATED_PATH,
            mimetype="text/csv",
            as_attachment=True,
            download_name="terjemahan_steam_reviews.csv",
        )
    except Exception as e:
        return jsonify(
            {"status": "error", "message": f"Gagal menyimpan file: {e}"}
        ), 500


@terjemahan_bp.route("/download")
def download_latest():
    if os.path.exists(TRANSLATED_PATH):
        return send_file(
            TRANSLATED_PATH,
            as_attachment=True,
            download_name="terjemahan_steam_reviews.csv",
        )
    return jsonify({"error": "File hasil terjemahan tidak ditemukan."}), 404


@terjemahan_bp.route("/")
def index():
    return render_template("terjemahan.html", page_name="terjemahan")

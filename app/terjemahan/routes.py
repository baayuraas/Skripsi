import os
import csv
import re
import time
import logging
from flask import Blueprint, request, jsonify, render_template, send_file
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor

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
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

translated_data = []
ignored_words = set()


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
    daftar_kata_ulang = []

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
            daftar_kata_ulang.append(f"{k1}-{k2}")
            return f"{k1} - {k2}"
        return f"{k1} {k2}"

    corrected = re.sub(r"\b(\w+)\s*-\s*(\w+)\b", koreksi, text)
    return corrected


def translate_large_text(text, target_language="id"):
    if not text or not isinstance(text, str) or text.strip() == "":
        return ""
    chunks = split_text_into_chunks(text, 5000)
    translated_chunks = []

    for chunk in chunks:
        attempts = 0
        while attempts < 3:
            try:
                translated = GoogleTranslator(
                    source="auto", target=target_language
                ).translate(chunk)
                translated = fix_hyphen_and_detect_reduplication(translated)
                translated_chunks.append(translated)
                break
            except Exception as e:
                attempts += 1
                logging.error(f"Error translating chunk: {e}")
                time.sleep(5)
    return " ".join(translated_chunks)


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
        translated_data = []
        rows = list(reader)
        cache = {}

        def translate_row(row):
            steamid = row.get("SteamID", "")
            ulasan = row.get("Ulasan") or ""
            status = row.get("Status", "")
            if not isinstance(ulasan, str):
                ulasan = ""
            if ulasan.strip() == "":
                terjemahan = ""
            elif ulasan in cache:
                terjemahan = cache[ulasan]
            else:
                terjemahan = translate_large_text(ulasan, "id")
                cache[ulasan] = terjemahan
            return {
                "SteamID": steamid,
                "Ulasan": ulasan,
                "Terjemahan": terjemahan,
                "Status": status,
            }

        with ThreadPoolExecutor(max_workers=5) as executor:
            translated_data = list(executor.map(translate_row, rows))

        return jsonify(
            {"message": "Berhasil diterjemahkan!", "data": translated_data}
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
        return send_file(TRANSLATED_PATH, as_attachment=True)
    return jsonify({"error": "File hasil terjemahan tidak ditemukan."}), 404


@terjemahan_bp.route("/")
def index():
    return render_template("terjemahan.html", page_name="terjemahan")

import re
import time
import logging
import os
import csv
from flask import Blueprint, request, jsonify, render_template, send_file
from deep_translator import GoogleTranslator

terjemahan_bp = Blueprint(
    "terjemahan",
    __name__,
    url_prefix="/terjemahan",
    template_folder="templates",
    static_folder="static",
)

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "terjemahan")
TRANSLATED_PATH = os.path.join(UPLOAD_FOLDER, "terjemahan_steam_reviews.csv")

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
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > max_chars:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += " " + sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def fix_hyphen_spacing_conditionally(text):
    def replacer(match):
        a, b = match.group(1), match.group(2)
        if a.lower() == b.lower():
            return f"{a} - {b}"
        return f"{a}-{b}"

    return re.sub(r"\b(\w+)-(\w+)\b", replacer, text)


def translate_large_text(text, target_language="id"):
    chunks = split_text_into_chunks(text, 5000)
    translated_chunks = []

    for chunk in chunks:
        attempts = 0
        while attempts < 3:
            try:
                translated_chunk = GoogleTranslator(
                    source="auto", target=target_language
                ).translate(chunk)
                translated_chunk = fix_hyphen_spacing_conditionally(translated_chunk)
                translated_chunks.append(translated_chunk)
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
        if "file" not in request.files:
            return jsonify({"error": "Tidak ada file yang diunggah."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Nama file kosong."}), 400

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

        for row in reader:
            steamid = row["SteamID"]
            ulasan = row["Ulasan"]
            status = row["Status"]
            terjemahan = translate_large_text(ulasan, "id")
            translated_data.append(
                {
                    "SteamID": steamid,
                    "Ulasan": ulasan,
                    "Terjemahan": terjemahan,
                    "Status": status,
                }
            )

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

    # ✅ Buat folder jika belum ada
    os.makedirs(os.path.dirname(TRANSLATED_PATH), exist_ok=True)

    try:
        with open(TRANSLATED_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["SteamID", "Ulasan", "Terjemahan", "Status"])
            for review in translated_data:
                formatted_review = (
                    review["Ulasan"].replace("\n", " ").replace("\r", " ")
                )
                writer.writerow(
                    [
                        review["SteamID"],
                        formatted_review,
                        review["Terjemahan"],
                        review["Status"],
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
    return render_template("terjemahan.html")

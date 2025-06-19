from flask import Blueprint, request, jsonify, render_template, send_file
import csv
import pandas as pd
import re
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from markupsafe import escape
import werkzeug.exceptions

prepro_bp = Blueprint(
    "prepro",
    __name__,
    url_prefix="/prepro",
    template_folder="templates",
    static_folder="static",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "preproses")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
PREPRO_CSV_PATH = os.path.join(UPLOAD_FOLDER, "processed_data.csv")

# Sastrawi + Stopwords
stopword_factory = StopWordRemoverFactory()
stemmer = StemmerFactory().create_stemmer()
stop_words = set(stopword_factory.get_stop_words()).union(
    set(stopwords.words("indonesian"))
)

# File tambahan
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "stopword_list.txt"), "r", encoding="utf-8") as f:
    stop_words.update(f.read().splitlines())

normalisasi_dict = {}
with open(os.path.join(BASE_DIR, "normalisasi_list.txt"), "r", encoding="utf-8") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            normalisasi_dict[k.strip().lower()] = v.strip().lower()

game_terms = {
    "archon",
    "bot",
    "creep",
    "crownfall",
    "icefrog",
    "meta",
    "overwatch",
    "pudge",
    "spam",
    "valve",
    "warcraft",
    "willow",
}
kata_tidak_relevan = {"f4s", "bllyat", "crownfall", "groundhog", "qith", "mook"}
CHUNK_SIZE = 500
MAX_FILE_SIZE = 2 * 1024 * 1024


def hapus_kata_ulang(word):
    word = re.sub(r"\b(\w+)-\1\b", r"\1", word)
    word = re.sub(r"\b(\w{3,}?)(?:\1)\b", r"\1", word)
    word = re.sub(r"\b(?:ber|se|ter|me|di|ke|pe|per)?(\w{3,}?)(?:\1)\b", r"\1", word)
    return word


def bersihkan_terjemahan(teks: str) -> str:
    if pd.isna(teks) or not isinstance(teks, str):
        return ""
    teks = re.sub(r"\[.*?\]", " ", teks)
    teks = re.sub(r"http\S+", " ", teks)
    teks = re.sub(r"\b\d+[kK]?\b", " ", teks)
    teks = re.sub(r"\b\w+-\w+\b", lambda m: m.group(0).replace("-", " "), teks)
    teks = re.sub(r"[^a-zA-Z0-9\s-]", " ", teks)
    teks = re.sub(r"\b[a-zA-Z]{1,3}\b", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    return teks


def normalisasi_teks(words):
    return [
        normalisasi_dict.get(hapus_kata_ulang(w).lower(), w) for w in words if w.strip()
    ]


def hapus_stopword(words):
    return [
        w
        for w in words
        if (w not in stop_words and w not in kata_tidak_relevan) or w in game_terms
    ]


def stemming_teks(words):
    return [stemmer.stem(w) for w in words]


def register_template_filters(app):
    app.jinja_env.filters["sanitize"] = escape


@prepro_bp.route("/preproses", methods=["POST"])
def preproses():
    try:
        file = request.files.get("file")
        if not file or not file.filename or not file.filename.endswith(".csv"):
            return jsonify(
                {"error": "Format file tidak valid, hanya mendukung CSV."}
            ), 400
        if file.content_length > MAX_FILE_SIZE:
            return jsonify({"error": "Ukuran file melebihi 2 MB."}), 400

        chunks = pd.read_csv(file.stream, chunksize=CHUNK_SIZE)
        processed = []

        for chunk in chunks:
            if "Terjemahan" not in chunk.columns:
                return jsonify({"error": "Kolom 'Terjemahan' tidak ditemukan."}), 400

            chunk["Terjemahan"] = chunk["Terjemahan"].fillna("")
            chunk["Clean Data"] = chunk["Terjemahan"].apply(bersihkan_terjemahan)
            chunk["Case Folding"] = chunk["Clean Data"].str.lower()
            chunk["Tokenisasi"] = chunk["Case Folding"].apply(word_tokenize)
            chunk["Stopword"] = chunk["Tokenisasi"].apply(hapus_stopword)
            chunk["Normalisasi"] = chunk["Stopword"].apply(normalisasi_teks)
            chunk["Stemming"] = chunk["Normalisasi"].apply(stemming_teks)
            chunk["Hasil"] = chunk["Stemming"].apply(lambda x: " ".join(x))

            if "Status" not in chunk.columns:
                chunk["Status"] = ""

            processed.append(chunk)

        result_df = pd.concat(processed, ignore_index=True)
        return jsonify(
            {"message": "Proses berhasil!", "data": result_df.to_dict(orient="records")}
        )

    except pd.errors.EmptyDataError:
        return jsonify({"error": "File CSV kosong atau tidak dapat dibaca."}), 400
    except werkzeug.exceptions.RequestEntityTooLarge:
        return jsonify({"error": "Ukuran file terlalu besar."}), 413
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@prepro_bp.route("/save_csv", methods=["POST"])
def save_csv():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Tidak ada data untuk disimpan."}), 400

        df = pd.DataFrame(data)

        expected_cols = [
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
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ""

        df = df.reindex(columns=expected_cols, fill_value="")

        for col in df.columns:
            df[col] = df[col].apply(
                lambda val: " ".join(val) if isinstance(val, list) else str(val)
            )

        os.makedirs(os.path.dirname(PREPRO_CSV_PATH), exist_ok=True)

        # Simpan file terlebih dahulu
        df.to_csv(
            PREPRO_CSV_PATH, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig"
        )

        # Kirim file menggunakan send_file
        return send_file(
            PREPRO_CSV_PATH,
            mimetype="text/csv",
            as_attachment=True,
            download_name="processed_data.csv",
        )

    except Exception as e:
        print(">> ERROR:", str(e))
        return jsonify({"error": f"Gagal menyimpan: {str(e)}"}), 500


@prepro_bp.route("/download")
def download_preprocessed():
    if os.path.exists(PREPRO_CSV_PATH):
        return send_file(PREPRO_CSV_PATH, as_attachment=True)
    return jsonify({"error": "File hasil preproses tidak ditemukan."}), 404


@prepro_bp.route("/")
def index():
    return render_template("preprosessing.html")

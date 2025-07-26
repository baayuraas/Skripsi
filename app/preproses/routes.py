import csv
import json
import os
import re
import time
import pandas as pd
import emoji
import unicodedata
from flask import Blueprint, request, jsonify, render_template, send_file
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from markupsafe import escape
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from deep_translator import GoogleTranslator, exceptions as dt_exceptions
from werkzeug.exceptions import RequestEntityTooLarge
from multiprocessing.pool import ThreadPool

prepro_bp = Blueprint(
    "prepro",
    __name__,
    url_prefix="/prepro",
    template_folder="templates",
    static_folder="static",
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "preproses")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
PREPRO_CSV_PATH = os.path.join(UPLOAD_FOLDER, "processed_data.csv")
TXT_DIR = os.path.dirname(os.path.abspath(__file__))
stopword_factory = StopWordRemoverFactory()
stemmer = StemmerFactory().create_stemmer()
stop_words = set(stopword_factory.get_stop_words()).union(
    set(stopwords.words("indonesian"))
)

with open(os.path.join(TXT_DIR, "stopword_list.txt"), "r", encoding="utf-8") as f:
    stop_words.update(f.read().splitlines())

normalisasi_dict = {}
with open(os.path.join(TXT_DIR, "normalisasi_list.txt"), "r", encoding="utf-8") as f:
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
terjemahan_cache = {}


def register_template_filters(app):
    app.jinja_env.filters["sanitize"] = escape


def bersihkan_terjemahan(teks: str) -> str:
    if pd.isna(teks) or not isinstance(teks, str):
        return ""
    teks = emoji.replace_emoji(teks, replace=" ")
    teks = re.sub(r"[-–—…“”‘’«»]", " ", teks)
    teks = re.sub(r"\[.*?\]", " ", teks)
    teks = re.sub(r"http\S+", " ", teks)
    teks = re.sub(r"\b[\d\w]*\d[\w\d]*\b", " ", teks)
    teks = re.sub(r"[^a-zA-Z0-9\s]", " ", teks)
    teks = re.sub(r"\b[a-zA-Z]{1,3}\b", " ", teks)
    teks = unicodedata.normalize("NFKD", teks).encode("ascii", "ignore").decode("utf-8")
    return re.sub(r"\s+", " ", teks).strip()


def hapus_kata_ulang(word):
    return re.sub(r"\b(\w{3,}?)(?:\1)\b", r"\1", word)


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


def deteksi_bukan_indonesia(words: list) -> bool:
    try:
        return detect(" ".join(words)) != "id"
    except LangDetectException:
        return False


def terjemahkan_ke_indonesia(words: list) -> list:
    global terjemahan_cache
    joined = " ".join(words).strip().lower()
    if not joined:
        return words
    if joined in terjemahan_cache:
        return word_tokenize(terjemahan_cache[joined])

    for _ in range(3):
        try:
            hasil = GoogleTranslator(source="auto", target="id").translate(joined)
            terjemahan_cache[joined] = hasil
            return word_tokenize(hasil.lower())
        except (dt_exceptions.NotValidPayload, dt_exceptions.TranslationNotFound):
            return words
        except Exception:
            time.sleep(1)
    return words


def proses_baris_aman(terjemahan):
    try:
        if pd.isna(terjemahan) or not isinstance(terjemahan, str):
            return ["", "", [], [], [], [], ""]

        clean = bersihkan_terjemahan(terjemahan)
        folded = clean.lower()
        token = word_tokenize(folded)
        stop = hapus_stopword(token)
        norm = normalisasi_teks(stop)
        stem = stemming_teks(norm)
        hasil = " ".join(stem)

        # Cek hasil akhir saja
        if hasil.strip() and deteksi_bukan_indonesia([hasil]):
            print(">> Translate ulang karena hasil masih bukan ID.")
            translated = GoogleTranslator(source="auto", target="id").translate(clean).lower()
            token = word_tokenize(translated)
            stop = hapus_stopword(token)
            norm = normalisasi_teks(stop)
            stem = stemming_teks(norm)
            hasil = " ".join(stem)

        return [clean, folded, token, stop, norm, stem, hasil]
    except Exception:
        return ["", "", [], [], [], [], ""]


@prepro_bp.route("/preproses", methods=["POST"])
def preproses():
    try:
        file = request.files.get("file")
        if not file or not file.filename or not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Format file tidak valid, hanya mendukung CSV."}), 400
        if file.content_length > MAX_FILE_SIZE:
            return jsonify({"error": "Ukuran file melebihi 2 MB."}), 400

        chunks = pd.read_csv(file.stream, chunksize=CHUNK_SIZE)
        processed = []
        pool = ThreadPool(4)

        for chunk in chunks:
            if "Terjemahan" not in chunk.columns:
                return jsonify({"error": "Kolom 'Terjemahan' tidak ditemukan."}), 400
            chunk["Terjemahan"] = chunk["Terjemahan"].fillna("")
            hasil_list = pool.map(proses_baris_aman, chunk["Terjemahan"].tolist())
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
            chunk = pd.concat([chunk.reset_index(drop=True), hasil_df], axis=1)
            if "Status" not in chunk.columns:
                chunk["Status"] = ""
            processed.append(chunk)

        result_df = pd.concat(processed, ignore_index=True)

        df_for_file = result_df.copy(deep=True)
        list_cols = ["Tokenisasi", "Stopword", "Normalisasi", "Stemming"]
        for col in list_cols:
            df_for_file[col] = df_for_file[col].apply(
                lambda val: json.dumps(val, ensure_ascii=False) if isinstance(val, list) else "[]"
            )
        for col in df_for_file.columns:
            if col not in list_cols:
                df_for_file[col] = df_for_file[col].astype(str).replace({'"': '""'})

        os.makedirs(os.path.dirname(PREPRO_CSV_PATH), exist_ok=True)
        df_for_file.to_csv(
            PREPRO_CSV_PATH,
            index=False,
            quoting=csv.QUOTE_ALL,
            encoding="utf-8-sig",
            lineterminator="\n",
            doublequote=True,
        )

        return jsonify({
            "message": "Preprocessing selesai dan disimpan!",
            "data": result_df.to_dict(orient="records")
        })

    except pd.errors.EmptyDataError:
        return jsonify({"error": "File CSV kosong atau tidak dapat dibaca."}), 400
    except RequestEntityTooLarge:
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

        list_cols = ["Tokenisasi", "Stopword", "Normalisasi", "Stemming"]
        for col in list_cols:
            df[col] = df[col].apply(
                lambda val: json.dumps(val, ensure_ascii=False) if isinstance(val, list) else "[]"
            )

        for col in df.columns:
            if col not in list_cols:
                df[col] = df[col].astype(str).replace({'"': '""'})

        os.makedirs(os.path.dirname(PREPRO_CSV_PATH), exist_ok=True)
        df.to_csv(
            PREPRO_CSV_PATH,
            index=False,
            quoting=csv.QUOTE_ALL,
            encoding="utf-8-sig",
            lineterminator="\n",
            doublequote=True,
        )

        return send_file(
            PREPRO_CSV_PATH,
            mimetype="text/csv",
            as_attachment=True,
            download_name="processed_data.csv",
        )

    except Exception as e:
        return jsonify({"error": f"Gagal menyimpan: {str(e)}"}), 500


@prepro_bp.route("/download")
def download_preprocessed():
    if os.path.exists(PREPRO_CSV_PATH):
        return send_file(PREPRO_CSV_PATH, as_attachment=True)
    return jsonify({"error": "File hasil preproses tidak ditemukan."}), 404


@prepro_bp.route("/")
def index():
    return render_template("preprosessing.html", page_name="prepro"), 200

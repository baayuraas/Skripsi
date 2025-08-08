import csv
import json
import os
import re
import logging
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
from functools import lru_cache
from typing import List, Dict, Tuple, Union

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
MAX_FILE_SIZE = 2 * 1024 * 1024
CHUNK_SIZE = 500
CACHE_FILE = os.path.join(BASE_DIR, "cache_translate.json")
CACHE_SIZE_LIMIT = 10000

stopword_factory = StopWordRemoverFactory()
stop_words = set(stopword_factory.get_stop_words()).union(
    set(stopwords.words("indonesian"))
)
logger.info("Komponen NLP diinisialisasi") 

# Load custom stopwords
try:
    with open(os.path.join(TXT_DIR, "stopword_list.txt"), "r", encoding="utf-8") as f:
        stop_words.update(f.read().splitlines())
    logger.info("Custom stopwords loaded successfully")
except Exception as e:
    logger.error(f"Failed to load custom stopwords: {e}")
    raise

normalisasi_dict = {}
try:
    with open(os.path.join(TXT_DIR, "normalisasi_list.txt"), "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                normalisasi_dict[k.strip().lower()] = v.strip().lower()
    logger.info("Normalization dictionary loaded successfully")
except Exception as e:
    logger.error(f"Failed to load normalization dictionary: {e}")
    raise

game_terms = {
    "archon", "bot", "creep", "crownfall", "icefrog", "meta", 
    "overwatch", "pudge", "spam", "valve", "warcraft", "willow"
}
irrelevant_words = {"f4s", "bllyat", "crownfall", "groundhog", "qith", "mook"}

# Translation cache with size limit
terjemahan_cache = {}

if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            terjemahan_cache = json.load(f)
            # Enforce cache size limit
            if len(terjemahan_cache) > CACHE_SIZE_LIMIT:
                terjemahan_cache = dict(list(terjemahan_cache.items())[:CACHE_SIZE_LIMIT])
        logger.info(f"Loaded translation cache with {len(terjemahan_cache)} entries")
    except Exception as e:
        logger.error(f"Failed to load translation cache: {e}")
        terjemahan_cache = {}

def simpan_cache_ke_file():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(terjemahan_cache, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved translation cache with {len(terjemahan_cache)} entries")
    except Exception as e:
        logger.error(f"Failed to save translation cache: {e}")

def validate_input_file(file) -> Tuple[bool, Union[str, None]]:
    if not file or not file.filename:
        return False, "No file provided"
    
    if not file.filename.lower().endswith(".csv"):
        return False, "Only CSV files are supported"
    
    if file.content_length > MAX_FILE_SIZE:
        return False, f"File size exceeds {MAX_FILE_SIZE/1024/1024}MB limit"
    
    return True, None

def bersihkan_terjemahan(teks: str) -> str:
    if pd.isna(teks) or not isinstance(teks, str):
        return ""
    
    try:
        teks = emoji.replace_emoji(teks, replace=" ")
        teks = re.sub(r"[-–—…“”‘’«»]", " ", teks)
        teks = re.sub(r"\[.*?\]", " ", teks)
        teks = re.sub(r"http\S+", " ", teks)
        teks = re.sub(r"\b[\d\w]*\d[\w\d]*\b", " ", teks)
        teks = re.sub(r"[^a-zA-Z0-9\s]", " ", teks)
        teks = re.sub(r"\b[a-zA-Z]{1,3}\b", " ", teks)
        teks = unicodedata.normalize("NFKD", teks).encode("ascii", "ignore").decode("utf-8")
        
        return re.sub(r"\s+", " ", teks).strip()
    
    except Exception as e:
        logger.error(f"Error in bersihkan_terjemahan: {e}")
        return ""

@lru_cache(maxsize=1000)
def hapus_kata_ulang(word: str) -> str:
    try:
        return re.sub(r"\b(\w{3,}?)(?:\1)\b", r"\1", word)
    except Exception as e:
        logger.error(f"Error in hapus_kata_ulang: {e}")
        return word

def normalisasi_teks(words: List[str]) -> List[str]:
    try:
        return [
            normalisasi_dict.get(hapus_kata_ulang(w).lower(), w) 
            for w in words 
            if w.strip()
        ]
    except Exception as e:
        logger.error(f"Error in normalisasi_teks: {e}")
        return words

def hapus_stopword(words: List[str]) -> List[str]:
    try:
        return [
            w for w in words
            if (w not in stop_words and w not in irrelevant_words) or w in game_terms
        ]
    except Exception as e:
        logger.error(f"Error in hapus_stopword: {e}")
        return words

def stemming_teks(words: List[str]) -> List[str]:
    try:
        return [stemmer.stem(w) for w in words]
    except Exception as e:
        logger.error(f"Error in stemming_teks: {e}")
        return words

def translate_batch_cached(kata_non_id: List[str]) -> None:
    if not kata_non_id:
        return
    
    try:
        kata_belum_diterjemahkan = [
            w for w in kata_non_id 
            if w not in terjemahan_cache and len(w) > 2
        ]
        
        if not kata_belum_diterjemahkan:
            return
        
        logger.info(f"Translating batch of {len(kata_belum_diterjemahkan)} words")
        
        batch_size = 50
        for i in range(0, len(kata_belum_diterjemahkan), batch_size):
            batch = kata_belum_diterjemahkan[i:i + batch_size]
            
            try:
                translator = GoogleTranslator(source="auto", target="id")
                hasil_batch = translator.translate_batch(batch)
                
                for asli, hasil in zip(batch, hasil_batch):
                    if hasil:
                        terjemahan_cache[asli] = hasil.lower()
                        logger.debug(f"Translated: {asli} → {hasil.lower()}")

                if i % 200 == 0:
                    simpan_cache_ke_file()

                time.sleep(1)

            except dt_exceptions.TooManyRequests:
                logger.warning("Translation API rate limit reached. Waiting...")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Translation error for batch {i}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error in translate_batch_cached: {e}")
    finally:
        simpan_cache_ke_file()

def proses_baris_aman(terjemahan: str) -> List[Union[str, List[str]]]:
    try:
        if pd.isna(terjemahan) or not isinstance(terjemahan, str):
            return ["", "", [], [], [], [], ""]

        clean = bersihkan_terjemahan(terjemahan)
        if not clean:
            return ["", "", [], [], [], [], ""]
        folded = clean.lower()
        token = word_tokenize(folded)
        if not token:
            return [clean, folded, [], [], [], [], ""]
        kata_non_id = []
        for w in token:
            try:
                if len(w) <= 2:
                    continue
                if detect(w) != "id":
                    kata_non_id.append(w)
            except LangDetectException:
                continue

        if kata_non_id:
            translate_batch_cached(kata_non_id)
        hasil_token = []
        for w in token:
            if len(w) <= 2:
                continue
            if w in terjemahan_cache:
                hasil_token.extend(word_tokenize(terjemahan_cache[w]))
            else:
                hasil_token.append(w)
        stop = hapus_stopword(hasil_token)
        norm = normalisasi_teks(stop)
        stem = stemming_teks(norm)
        hasil = " ".join(stem)
        logger.debug(f"Processed: {terjemahan[:50]}... → {hasil[:50]}...")
        return [clean, folded, token, stop, norm, stem, hasil]

    except Exception as e:
        logger.error(f"Error in proses_baris_aman: {e}")
        return ["", "", [], [], [], [], ""]

@prepro_bp.route("/preproses", methods=["POST"])
def preproses():
    start_time = time.time()
    
    try:
        file = request.files.get("file")
        is_valid, error_msg = validate_input_file(file)
        if not is_valid:
            logger.error(f"Invalid file: {error_msg}")
            return jsonify({"error": error_msg}), 400

        chunks = pd.read_csv(file.stream, chunksize=CHUNK_SIZE)
        processed = []
        total_rows = 0
        with ThreadPool(4) as pool:
            for i, chunk in enumerate(chunks, start=1):
                logger.info(f"Processing chunk {i} with {len(chunk)} rows")
                
                if "Terjemahan" not in chunk.columns:
                    logger.error("'Terjemahan' column not found")
                    return jsonify({"error": "Kolom 'Terjemahan' tidak ditemukan."}), 400
                
                chunk["Terjemahan"] = chunk["Terjemahan"].fillna("")
                total_rows += len(chunk)

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

        df_for_file.drop(
            columns=[col for col in df_for_file.columns if col.lower() in {"ulasan", "terjemahan"}], 
            inplace=True, 
            errors="ignore"
        )
        
        os.makedirs(os.path.dirname(PREPRO_CSV_PATH), exist_ok=True)
        df_for_file.to_csv(
            PREPRO_CSV_PATH,
            index=False,
            quoting=csv.QUOTE_ALL,
            encoding="utf-8-sig",
            lineterminator="\n",
            doublequote=True,
        )
        
        simpan_cache_ke_file()

        processing_time = time.time() - start_time
        logger.info(
            f"Processed {total_rows} rows in {processing_time:.2f} seconds "
            f"({total_rows/processing_time:.2f} rows/sec)"
        )

        return jsonify({
            "message": "Preprocessing selesai dan disimpan!",
            "stats": {
                "total_rows": total_rows,
                "processing_time": f"{processing_time:.2f} seconds",
                "rows_per_second": f"{total_rows/processing_time:.2f}"
            },
            "data": result_df.head(50).to_dict(orient="records")  # Return first 50 rows only
        })

    except pd.errors.EmptyDataError:
        logger.error("Empty CSV file")
        return jsonify({"error": "File CSV kosong atau tidak dapat dibaca."}), 400
    except RequestEntityTooLarge:
        logger.error("File too large")
        return jsonify({"error": "Ukuran file terlalu besar."}), 413
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500

@prepro_bp.route("/save_csv", methods=["POST"])
def save_csv():
    try:
        data = request.json
        if not data:
            logger.error("No data provided for saving")
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

        simpan_cache_ke_file()

        logger.info("Successfully saved processed data to CSV")
        return send_file(
            PREPRO_CSV_PATH,
            mimetype="text/csv",
            as_attachment=True,
            download_name="processed_data.csv",
        )

    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        return jsonify({"error": f"Gagal menyimpan: {str(e)}"}), 500

@prepro_bp.route("/download")
def download_preprocessed():
    if os.path.exists(PREPRO_CSV_PATH):
        logger.info("Serving processed CSV file")
        return send_file(PREPRO_CSV_PATH, as_attachment=True)
    
    logger.error("Processed file not found")
    return jsonify({"error": "File hasil preproses tidak ditemukan."}), 404

@prepro_bp.route("/")
def index():
    return render_template("preprosessing.html", page_name="prepro"), 200

def register_template_filters(app):
    app.jinja_env.filters["sanitize"] = escape
    logger.info("Template filters registered")
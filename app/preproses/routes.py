import csv
import json
import os
import re
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
from deep_translator import GoogleTranslator
from multiprocessing.pool import ThreadPool

# --- Konfigurasi dasar ---
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

# --- Load stemming_list.txt ---
stemming_dict = {}
stemming_file = os.path.join(TXT_DIR, "stemming_list.txt")
if os.path.exists(stemming_file):
    with open(stemming_file, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                stemming_dict[k.strip().lower()] = v.strip().lower()
    print(f"[STEMMING LIST] Loaded {len(stemming_dict)} pasangan dari stemming_list.txt")
else:
    print("[STEMMING LIST] File tidak ditemukan, gunakan default Sastrawi.")

with open(os.path.join(TXT_DIR, "game_term.txt"), "r", encoding="utf-8") as f:
    game_terms = set(line.strip().lower() for line in f if line.strip())

with open(os.path.join(TXT_DIR, "kata_tidak_relevan.txt"), "r", encoding="utf-8") as f:
    kata_tidak_relevan = set(line.strip().lower() for line in f if line.strip())

CHUNK_SIZE = 500
MAX_FILE_SIZE = 2 * 1024 * 1024
CACHE_FILE = os.path.join(BASE_DIR, "cache_translate.json")

# --- Cache translate ---
terjemahan_cache = {}
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            terjemahan_cache = json.load(f)
    except Exception:
        terjemahan_cache = {}

def simpan_cache_ke_file():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(terjemahan_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR simpan cache]: {e}")

def translate_long_text(text, source="auto", target="id", max_len=5000):
    translator = GoogleTranslator(source=source, target=target)
    hasil = []

    # Pecah teks berdasarkan tanda baca atau newline
    kalimat = re.split(r"(?<=[.!?]) +|\n", text)

    buffer = ""
    for k in kalimat:
        if len(buffer) + len(k) < max_len:
            buffer += " " + k
        else:
            if buffer.strip():
                hasil.append(translator.translate(buffer.strip()))
            buffer = k

    if buffer.strip():
        hasil.append(translator.translate(buffer.strip()))

    return " ".join(hasil)

# --- Utility ---
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
    hasil = []
    for w in words:
        wl = w.lower()
        if wl in normalisasi_dict:
            mapped = normalisasi_dict[wl]
            hasil.extend(mapped.split())
        else:
            hasil.append(wl)
    return hasil

def hapus_stopword(words):
    return [
        w
        for w in words
        if (w not in stop_words and w not in kata_tidak_relevan) or w in game_terms
    ]

def stemming_teks(words):
    hasil = []
    for w in words:
        wl = w.lower()
        if wl in stemming_dict:
            mapped = stemming_dict[wl]
            hasil.extend(mapped.split())
        else:
            hasil.append(stemmer.stem(wl))
    return hasil

def deteksi_bukan_indonesia(words: list) -> bool:
    try:
        return detect(" ".join(words)) != "id"
    except LangDetectException:
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

# --- PERBAIKAN: Translasi batch kata yang lebih baik ---
def translate_batch_cached(kata_non_id):
    # Filter kata yang belum ada di cache, bukan game term, bukan kata ambigu
    kata_belum_diterjemahkan = [
        w for w in kata_non_id
        if w not in terjemahan_cache
        and w not in game_terms
        and w not in kata_id_pasti
        and len(w) > 2  # hanya kata dengan panjang > 2
    ]

    if not kata_belum_diterjemahkan:
        return

    try:
        # Batch translation dengan chunking
        chunk_size = 50  # Google Translate batch limit
        for i in range(0, len(kata_belum_diterjemahkan), chunk_size):
            chunk = kata_belum_diterjemahkan[i:i + chunk_size]
            hasil_batch = GoogleTranslator(source="auto", target="id").translate_batch(chunk)
            
            for asli, hasil in zip(chunk, hasil_batch):
                if hasil and hasil.strip():  # hanya simpan jika hasil valid
                    terjemahan_cache[asli] = hasil.lower()  # simpan dalam lowercase
                    
    except Exception as e:
        print(f"[ERROR] Gagal translate batch: {e}")

# --- PERBAIKAN: Deteksi kata non-ID dengan whitelist yang lebih baik ---
def deteksi_kata_non_indonesia(words: list) -> list:
    kata_non_id = []
    try:
        for w in words:
            wl = w.lower()
            # Skip kata pendek, kata game, dan kata whitelist
            if (len(wl) <= 2 or 
                wl in kata_id_pasti or 
                wl in game_terms or
                wl in normalisasi_dict or  # kata yang sudah dinormalisasi
                wl in stemming_dict):     # kata yang sudah di-stem
                continue
            
            try:
                # Deteksi dengan confidence check
                lang = detect(wl)
                if lang != "id":
                    kata_non_id.append(wl)
            except LangDetectException:
                continue
    except Exception:
        return []
    return kata_non_id

# --- PERBAIKAN: Proses per baris dengan handling error yang lebih baik ---
def proses_baris_aman(terjemahan):
    try:
        # Validasi input
        if pd.isna(terjemahan) or not isinstance(terjemahan, str) or not terjemahan.strip():
            return ["", "", [], [], [], [], ""]

        clean = bersihkan_terjemahan(terjemahan)
        if not clean.strip():
            return ["", "", [], [], [], [], ""]
            
        folded = clean.lower()
        token = word_tokenize(folded)

        # Deteksi kata non-ID dengan filter yang lebih baik
        kata_non_id = deteksi_kata_non_indonesia(token)
        
        # Translate batch dengan filter
        translate_batch_cached(kata_non_id)

        # Proses token dengan terjemahan
        hasil_token = []
        for w in token:
            wl = w.lower()
            if wl in terjemahan_cache:
                translated_text = terjemahan_cache[wl]
                if translated_text and translated_text.strip():
                    translated_tokens = word_tokenize(translated_text.lower())
                    hasil_token.extend(translated_tokens)
                else:
                    hasil_token.append(wl)
            else:
                hasil_token.append(wl)

        # Proses NLP
        stop = hapus_stopword(hasil_token)
        norm = normalisasi_teks(stop)
        stem = stemming_teks(norm)
        hasil = " ".join(stem)

        # Validasi hasil akhir
        if not hasil.strip():
            # Fallback ke case folding
            hasil = folded

        return [clean, folded, token, stop, norm, stem, hasil]
        
    except Exception as e:
        print(f"[ERROR hybrid]: {e}")
        # Return minimal data instead of empty
        clean = bersihkan_terjemahan(terjemahan) if isinstance(terjemahan, str) else ""
        folded = clean.lower() if clean else ""
        return [clean, folded, [], [], [], [], folded]

# --- PERBAIKAN: Fungsi save_csv dengan handling empty data yang lebih baik ---
def save_csv(df: pd.DataFrame, file_path: str = PREPRO_CSV_PATH):
    """Menyimpan DataFrame ke CSV dengan format konsisten."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        df_for_file = df.copy(deep=True)

        # Konversi kolom list menjadi string JSON
        list_cols = ["Tokenisasi", "Stopword", "Normalisasi", "Stemming"]
        for col in list_cols:
            if col in df_for_file.columns:
                df_for_file[col] = df_for_file[col].apply(
                    lambda val: json.dumps(val, ensure_ascii=False)
                    if isinstance(val, list) and val
                    else "[]"
                )

        # Hapus kolom sensitif / tidak diperlukan
        df_for_file.drop(
            columns=[
                col for col in df_for_file.columns
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
            "Status"
        ]
        
        # Pastikan hanya ambil kolom yang memang ada
        kolom_ada = [col for col in kolom_urutan if col in df_for_file.columns]
        df_for_file = df_for_file[kolom_ada]

        # Simpan CSV
        df_for_file.to_csv(
            file_path,
            index=False,
            quoting=csv.QUOTE_ALL,
            encoding="utf-8-sig",
            lineterminator="\n",
            doublequote=True,
        )

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

        chunks = pd.read_csv(file.stream, chunksize=CHUNK_SIZE)
        processed = []
        pool = ThreadPool(4)

        for i, chunk in enumerate(chunks, start=1):
            print(f"[CHUNK {i}] Memproses {len(chunk)} baris...")
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

        # Simpan CSV otomatis
        save_csv(result_df)
        simpan_cache_ke_file()

        return jsonify(
            {
                "message": "Preprocessing selesai dan disimpan!",
                "data": result_df.to_dict(orient="records"),
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

@prepro_bp.route("/")
def index():
    return render_template("preprosessing.html", page_name="prepro"), 200
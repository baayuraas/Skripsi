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

# Konfigurasi logging yang lebih komprehensif
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("preprocessing.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Inisialisasi Blueprint Flask
prepro_bp = Blueprint(
    "prepro",
    __name__,
    url_prefix="/prepro",
    template_folder="templates",
    static_folder="static",
)

# Konfigurasi path dengan penanganan error
try:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "preproses")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    PREPRO_CSV_PATH = os.path.join(UPLOAD_FOLDER, "processed_data.csv")
    TXT_DIR = os.path.dirname(os.path.abspath(__file__))
    logger.info("Direktori dan path berhasil diinisialisasi")
except Exception as e:
    logger.error(f"Gagal menginisialisasi path: {str(e)}")
    raise

# Konstanta
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB
CHUNK_SIZE = 500
CACHE_FILE = os.path.join(BASE_DIR, "cache_translate.json")
CACHE_SIZE_LIMIT = 10000  # Batas ukuran cache

# Inisialisasi komponen NLP dengan penanganan error
try:
    stopword_factory = StopWordRemoverFactory()
    stemmer_factory = StemmerFactory()
    stemmer = (
        stemmer_factory.create_stemmer()
    )  # Inisialisasi stemmer yang sebelumnya hilang

    # Gabungkan stopwords dari berbagai sumber
    stop_words = set(stopword_factory.get_stop_words()).union(
        set(stopwords.words("indonesian"))
    )

    # Tambahkan stopwords kustom
    custom_stopwords = [
        "yg",
        "dg",
        "rt",
        "dgn",
        "ny",
        "d",
        "klo",
        "kalo",
        "amp",
        "biar",
        "bikin",
        "bilang",
        "gak",
        "ga",
        "krn",
        "nya",
        "nih",
        "sih",
        "si",
        "tau",
        "tdk",
        "tuh",
        "utk",
        "ya",
        "jd",
        "jgn",
        "sdh",
        "aja",
        "n",
        "t",
        "nyg",
        "hehe",
        "pen",
        "u",
        "nan",
        "loh",
        "rt",
        "&amp",
        "yah",
    ]
    stop_words.update(custom_stopwords)

    logger.info("Komponen NLP berhasil diinisialisasi")
except Exception as e:
    logger.error(f"Gagal inisialisasi komponen NLP: {str(e)}")
    raise

# Memuat stopwords kustom dengan penanganan error yang lebih baik
try:
    stopword_file = os.path.join(TXT_DIR, "stopword_list.txt")
    if os.path.exists(stopword_file):
        with open(stopword_file, "r", encoding="utf-8") as f:
            stop_words.update(f.read().splitlines())
        logger.info("Stopwords kustom berhasil dimuat")
    else:
        logger.warning(f"File stopword_list.txt tidak ditemukan di {stopword_file}")
except Exception as e:
    logger.error(f"Gagal memuat stopwords kustom: {str(e)}")

# Memuat kamus normalisasi
normalisasi_dict = {}
try:
    normalisasi_file = os.path.join(TXT_DIR, "normalisasi_list.txt")
    if os.path.exists(normalisasi_file):
        with open(normalisasi_file, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    normalisasi_dict[k.strip().lower()] = v.strip().lower()
        logger.info("Kamus normalisasi berhasil dimuat")
    else:
        logger.warning(
            f"File normalisasi_list.txt tidak ditemukan di {normalisasi_file}"
        )
except Exception as e:
    logger.error(f"Gagal memuat kamus normalisasi: {str(e)}")

# Istilah khusus domain
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

# Cache terjemahan dengan penanganan error yang lebih baik
terjemahan_cache = {}
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            terjemahan_cache = json.load(f)
            if len(terjemahan_cache) > CACHE_SIZE_LIMIT:
                terjemahan_cache = dict(
                    list(terjemahan_cache.items())[:CACHE_SIZE_LIMIT]
                )
        logger.info(f"Cache terjemahan dimuat dengan {len(terjemahan_cache)} entri")
    except Exception as e:
        logger.error(f"Gagal memuat cache terjemahan: {str(e)}")
        terjemahan_cache = {}


def simpan_cache_ke_file():
    """Menyimpan cache terjemahan ke file dengan penanganan error"""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(terjemahan_cache, f, ensure_ascii=False, indent=2)
        logger.info(f"Cache terjemahan disimpan dengan {len(terjemahan_cache)} entri")
    except Exception as e:
        logger.error(f"Gagal menyimpan cache terjemahan: {str(e)}")


def validasi_file_input(file) -> Tuple[bool, Union[str, None]]:
    """Validasi file input sebelum diproses"""
    if file is None:
        return False, "Tidak ada file yang diberikan"

    if not hasattr(file, "filename") or not file.filename:
        return False, "Nama file tidak valid"

    if not file.filename.lower().endswith(".csv"):
        return False, "Hanya file CSV yang didukung"

    if not hasattr(file, "stream"):
        return False, "File tidak memiliki atribut stream"

    if hasattr(file, "content_length") and file.content_length > MAX_FILE_SIZE:
        return False, f"Ukuran file melebihi batas {MAX_FILE_SIZE / 1024 / 1024}MB"

    return True, None


def bersihkan_teks(teks: str) -> str:
    """Membersihkan teks dari karakter tidak perlu"""
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
        teks = (
            unicodedata.normalize("NFKD", teks)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )
        return re.sub(r"\s+", " ", teks).strip()
    except Exception as e:
        logger.error(f"Error dalam bersihkan_teks: {str(e)}")
        return ""


@lru_cache(maxsize=1000)
def hapus_kata_ulang(kata: str) -> str:
    """Menghapus pengulangan kata dengan caching"""
    try:
        return re.sub(r"\b(\w{3,}?)(?:\1)\b", r"\1", kata)
    except Exception as e:
        logger.error(f"Error dalam hapus_kata_ulang: {str(e)}")
        return kata


def normalisasi_teks(kata_kata: List[str]) -> List[str]:
    """Normalisasi kata menggunakan kamus"""
    try:
        return [
            normalisasi_dict.get(hapus_kata_ulang(w).lower(), w)
            for w in kata_kata
            if w.strip()
        ]
    except Exception as e:
        logger.error(f"Error dalam normalisasi_teks: {str(e)}")
        return kata_kata


def hapus_stopword(kata_kata: List[str]) -> List[str]:
    """Menghapus stopword dengan pengecualian istilah khusus"""
    try:
        return [
            w
            for w in kata_kata
            if (w not in stop_words and w not in kata_tidak_relevan) or w in game_terms
        ]
    except Exception as e:
        logger.error(f"Error dalam hapus_stopword: {str(e)}")
        return kata_kata


def stemming_teks(kata_kata: List[str]) -> List[str]:
    """Melakukan stemming pada list kata"""
    try:
        return [stemmer.stem(w) for w in kata_kata]
    except Exception as e:
        logger.error(f"Error dalam stemming_teks: {str(e)}")
        return kata_kata


def terjemahkan_batch(kata_non_id: List[str]) -> None:
    """Menerjemahkan batch kata non-Indonesia dengan caching"""
    if not kata_non_id:
        return

    try:
        kata_baru = [w for w in kata_non_id if w not in terjemahan_cache and len(w) > 2]
        if not kata_baru:
            return

        logger.info(f"Menerjemahkan {len(kata_baru)} kata baru")

        ukuran_batch = 50
        for i in range(0, len(kata_baru), ukuran_batch):
            batch = kata_baru[i : i + ukuran_batch]

            try:
                penerjemah = GoogleTranslator(source="auto", target="id")
                hasil_batch = penerjemah.translate_batch(batch)

                for asli, hasil in zip(batch, hasil_batch):
                    if hasil:
                        terjemahan_cache[asli] = hasil.lower()
                        logger.debug(f"Diterjemahkan: {asli} → {hasil.lower()}")

                if i % 200 == 0:
                    simpan_cache_ke_file()

                time.sleep(1)  # Hindari rate limiting

            except dt_exceptions.TooManyRequests:
                logger.warning("Batas API terjemahan tercapai. Menunggu...")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error terjemahan batch {i}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error dalam terjemahkan_batch: {str(e)}")
    finally:
        simpan_cache_ke_file()


def proses_satu_baris(teks: str) -> List[Union[str, List[str]]]:
    """Pipeline pemrosesan untuk satu baris teks"""
    try:
        if pd.isna(teks) or not isinstance(teks, str):
            return ["", "", [], [], [], [], ""]

        # Langkah 1: Pembersihan teks
        bersih = bersihkan_teks(teks)
        if not bersih:
            return ["", "", [], [], [], [], ""]

        # Langkah 2: Case folding
        kecil = bersih.lower()

        # Langkah 3: Tokenisasi
        tokens = word_tokenize(kecil)
        if not tokens:
            return [bersih, kecil, [], [], [], [], ""]

        # Identifikasi kata non-Indonesia
        kata_asing = []
        for w in tokens:
            try:
                if len(w) <= 2:
                    continue
                if detect(w) != "id":
                    kata_asing.append(w)
            except LangDetectException:
                continue

        # Terjemahkan kata asing
        if kata_asing:
            terjemahkan_batch(kata_asing)

        # Gabungkan hasil terjemahan
        hasil_tokens = []
        for w in tokens:
            if len(w) <= 2:
                continue
            if w in terjemahan_cache:
                hasil_tokens.extend(word_tokenize(terjemahan_cache[w]))
            else:
                hasil_tokens.append(w)

        # Langkah 4: Stopword removal
        tanpa_stopword = hapus_stopword(hasil_tokens)

        # Langkah 5: Normalisasi
        dinormalisasi = normalisasi_teks(tanpa_stopword)

        # Langkah 6: Stemming
        distem = stemming_teks(dinormalisasi)

        # Hasil akhir
        hasil_akhir = " ".join(distem)

        logger.debug(f"Diproses: {teks[:50]}... → {hasil_akhir[:50]}...")
        return [
            bersih,
            kecil,
            tokens,
            tanpa_stopword,
            dinormalisasi,
            distem,
            hasil_akhir,
        ]

    except Exception as e:
        logger.error(f"Error dalam proses_satu_baris: {str(e)}")
        return ["", "", [], [], [], [], ""]


@prepro_bp.route("/preproses", methods=["POST"])
def preproses():
    """Endpoint utama untuk preprocessing"""
    waktu_mulai = time.time()

    try:
        file = request.files.get("file")
        valid, pesan_error = validasi_file_input(file)
        if not valid:
            logger.error(f"File tidak valid: {pesan_error}")
            return jsonify({"error": pesan_error}), 400

        # Proses file dalam chunk
        chunks = pd.read_csv(file.stream, chunksize=CHUNK_SIZE)
        hasil_proses = []
        total_baris = 0

        # Gunakan ThreadPool untuk pemrosesan paralel
        with ThreadPool(4) as pool:
            for i, chunk in enumerate(chunks, start=1):
                logger.info(f"Memproses chunk {i} dengan {len(chunk)} baris")

                if "Terjemahan" not in chunk.columns:
                    logger.error("Kolom 'Terjemahan' tidak ditemukan")
                    return jsonify({"error": "Kolom 'Terjemahan' tidak ditemukan"}), 400

                chunk["Terjemahan"] = chunk["Terjemahan"].fillna("")
                total_baris += len(chunk)

                # Proses setiap baris secara paralel
                hasil_list = pool.map(proses_satu_baris, chunk["Terjemahan"].tolist())

                # Gabungkan hasil
                df_hasil = pd.DataFrame(
                    hasil_list,
                    columns=[
                        "Teks Bersih",
                        "Case Folding",
                        "Tokenisasi",
                        "Tanpa Stopword",
                        "Normalisasi",
                        "Stemming",
                        "Hasil",
                    ],
                )

                chunk = pd.concat([chunk.reset_index(drop=True), df_hasil], axis=1)
                chunk["Status"] = chunk.get("Status", "")
                hasil_proses.append(chunk)

        # Gabungkan semua chunk
        df_final = pd.concat(hasil_proses, ignore_index=True)

        # Siapkan untuk ekspor CSV
        df_ekspor = df_final.copy()
        kolom_list = ["Tokenisasi", "Tanpa Stopword", "Normalisasi", "Stemming"]

        for kolom in kolom_list:
            df_ekspor[kolom] = df_ekspor[kolom].apply(
                lambda x: json.dumps(x, ensure_ascii=False)
                if isinstance(x, list)
                else "[]"
            )

        for kolom in df_ekspor.columns:
            if kolom not in kolom_list:
                df_ekspor[kolom] = df_ekspor[kolom].astype(str).replace({'"': '""'})

        # Hapus kolom mentah jika ada
        df_ekspor.drop(
            columns=[
                col
                for col in df_ekspor.columns
                if col.lower() in {"ulasan", "terjemahan"}
            ],
            inplace=True,
            errors="ignore",
        )

        # Simpan hasil
        os.makedirs(os.path.dirname(PREPRO_CSV_PATH), exist_ok=True)
        df_ekspor.to_csv(
            PREPRO_CSV_PATH,
            index=False,
            quoting=csv.QUOTE_ALL,
            encoding="utf-8-sig",
            lineterminator="\n",
            doublequote=True,
        )

        simpan_cache_ke_file()

        # Hitung statistik
        waktu_proses = time.time() - waktu_mulai
        logger.info(
            f"Memproses {total_baris} baris dalam {waktu_proses:.2f} detik "
            f"({total_baris / waktu_proses:.2f} baris/detik)"
        )

        return jsonify(
            {
                "pesan": "Preprocessing berhasil!",
                "statistik": {
                    "total_baris": total_baris,
                    "waktu_proses": f"{waktu_proses:.2f} detik",
                    "kecepatan": f"{total_baris / waktu_proses:.2f} baris/detik",
                },
                "contoh_data": df_final.head(50).to_dict(orient="records"),
            }
        )

    except pd.errors.EmptyDataError:
        logger.error("File CSV kosong")
        return jsonify({"error": "File CSV kosong atau tidak valid"}), 400
    except RequestEntityTooLarge:
        logger.error("File terlalu besar")
        return jsonify({"error": "Ukuran file melebihi batas 2MB"}), 413
    except Exception as e:
        logger.error(f"Error tak terduga: {str(e)}")
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500


@prepro_bp.route("/simpan_csv", methods=["POST"])
def simpan_csv():
    """Endpoint untuk menyimpan hasil preprocessing"""
    try:
        data = request.json
        if not data:
            logger.error("Tidak ada data untuk disimpan")
            return jsonify({"error": "Tidak ada data untuk disimpan"}), 400

        df = pd.DataFrame(data)

        kolom_wajib = [
            "SteamID",
            "Teks Bersih",
            "Case Folding",
            "Tokenisasi",
            "Tanpa Stopword",
            "Normalisasi",
            "Stemming",
            "Hasil",
            "Status",
        ]

        for kolom in kolom_wajib:
            if kolom not in df.columns:
                df[kolom] = ""

        df = df.reindex(columns=kolom_wajib, fill_value="")

        kolom_list = ["Tokenisasi", "Tanpa Stopword", "Normalisasi", "Stemming"]
        for kolom in kolom_list:
            df[kolom] = df[kolom].apply(
                lambda x: json.dumps(x, ensure_ascii=False)
                if isinstance(x, list)
                else "[]"
            )

        for kolom in df.columns:
            if kolom not in kolom_list:
                df[kolom] = df[kolom].astype(str).replace({'"': '""'})

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

        logger.info("Data berhasil disimpan ke CSV")
        return send_file(
            PREPRO_CSV_PATH,
            mimetype="text/csv",
            as_attachment=True,
            download_name="hasil_preprocessing.csv",
        )

    except Exception as e:
        logger.error(f"Gagal menyimpan CSV: {str(e)}")
        return jsonify({"error": f"Gagal menyimpan: {str(e)}"}), 500


@prepro_bp.route("/unduh")
def unduh_hasil():
    """Endpoint untuk mengunduh hasil preprocessing"""
    if os.path.exists(PREPRO_CSV_PATH):
        logger.info("Mengirim file hasil preprocessing")
        return send_file(PREPRO_CSV_PATH, as_attachment=True)

    logger.error("File hasil tidak ditemukan")
    return jsonify({"error": "File hasil preprocessing tidak ditemukan"}), 404


@prepro_bp.route("/")
def tampilan_utama():
    """Endpoint untuk tampilan utama"""
    return render_template("preprosessing.html", page_name="prepro"), 200


def daftarkan_filter_template(app):
    """Mendaftarkan filter template"""
    app.jinja_env.filters["saring"] = escape
    logger.info("Filter template terdaftar")

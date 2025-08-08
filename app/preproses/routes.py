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
from multiprocessing.pool import ThreadPool
from functools import lru_cache
from typing import List, Tuple, Union

# Konfigurasi logging dalam bahasa Indonesia
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
    logger.info("Path dan direktori berhasil diinisialisasi")
except Exception as e:
    logger.error(f"Gagal menginisialisasi path: {str(e)}")
    raise

# Konstanta
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB
CHUNK_SIZE = 500
CACHE_FILE = os.path.join(BASE_DIR, "cache_translate.json")
CACHE_SIZE_LIMIT = 10000

# Inisialisasi komponen NLP
try:
    stopword_factory = StopWordRemoverFactory()
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()  # Perbaikan: Inisialisasi stemmer

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

# Memuat stopwords kustom
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
normalization_dict = {}
try:
    normalization_file = os.path.join(TXT_DIR, "normalisasi_list.txt")
    if os.path.exists(normalization_file):
        with open(normalization_file, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    normalization_dict[k.strip().lower()] = v.strip().lower()
        logger.info("Kamus normalisasi berhasil dimuat")
    else:
        logger.warning(
            f"File normalisasi_list.txt tidak ditemukan di {normalization_file}"
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
irrelevant_words = {"f4s", "bllyat", "crownfall", "groundhog", "qith", "mook"}

# Cache terjemahan
translation_cache = {}
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            translation_cache = json.load(f)
            if len(translation_cache) > CACHE_SIZE_LIMIT:
                translation_cache = dict(
                    list(translation_cache.items())[:CACHE_SIZE_LIMIT]
                )
        logger.info(f"Cache terjemahan dimuat dengan {len(translation_cache)} entri")
    except Exception as e:
        logger.error(f"Gagal memuat cache terjemahan: {str(e)}")
        translation_cache = {}


def save_cache_to_file():
    """Menyimpan cache terjemahan ke file"""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(translation_cache, f, ensure_ascii=False, indent=2)
        logger.info(f"Cache terjemahan disimpan dengan {len(translation_cache)} entri")
    except Exception as e:
        logger.error(f"Gagal menyimpan cache terjemahan: {str(e)}")


def validate_input_file(file) -> Tuple[bool, Union[str, None]]:
    """Validasi file input"""
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


def clean_text(text: str) -> str:
    """Membersihkan teks dari karakter yang tidak diinginkan"""
    if pd.isna(text) or not isinstance(text, str):
        return ""

    try:
        text = emoji.replace_emoji(text, replace=" ")
        text = re.sub(r"[-–—…“”‘’«»]", " ", text)
        text = re.sub(r"\[.*?\]", " ", text)
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"\b[\d\w]*\d[\w\d]*\b", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\b[a-zA-Z]{1,3}\b", " ", text)
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )
        return re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        logger.error(f"Error dalam pembersihan teks: {str(e)}")
        return ""


@lru_cache(maxsize=1000)
def remove_repeated_words(word: str) -> str:
    """Menghapus pengulangan kata"""
    try:
        return re.sub(r"\b(\w{3,}?)(?:\1)\b", r"\1", word)
    except Exception as e:
        logger.error(f"Error menghapus pengulangan kata: {str(e)}")
        return word


def normalize_text(words: List[str]) -> List[str]:
    """Normalisasi teks menggunakan kamus"""
    try:
        return [
            normalization_dict.get(remove_repeated_words(w).lower(), w)
            for w in words
            if w.strip()
        ]
    except Exception as e:
        logger.error(f"Error normalisasi teks: {str(e)}")
        return words


def remove_stopwords(words: List[str]) -> List[str]:
    """Menghapus stopwords kecuali istilah khusus"""
    try:
        return [
            w
            for w in words
            if (w not in stop_words and w not in irrelevant_words) or w in game_terms
        ]
    except Exception as e:
        logger.error(f"Error menghapus stopwords: {str(e)}")
        return words


def stem_text(words: List[str]) -> List[str]:
    """Melakukan stemming pada kata"""
    try:
        return [stemmer.stem(w) for w in words]
    except Exception as e:
        logger.error(f"Error stemming: {str(e)}")
        return words


def translate_batch(non_id_words: List[str]) -> None:
    """Menerjemahkan kata non-Indonesia"""
    if not non_id_words:
        return

    try:
        new_words = [
            w for w in non_id_words if w not in translation_cache and len(w) > 2
        ]
        if not new_words:
            return

        logger.info(f"Memproses terjemahan {len(new_words)} kata baru")

        batch_size = 50
        for i in range(0, len(new_words), batch_size):
            batch = new_words[i : i + batch_size]

            try:
                translator = GoogleTranslator(source="auto", target="id")
                batch_results = translator.translate_batch(batch)

                for original, translated in zip(batch, batch_results):
                    if translated:
                        translation_cache[original] = translated.lower()
                        logger.debug(f"Terjemahan: {original} → {translated.lower()}")

                if i % 200 == 0:
                    save_cache_to_file()

                time.sleep(1)  # Menghindari rate limiting

            except dt_exceptions.TooManyRequests:
                logger.warning("Batas API terjemahan tercapai. Menunggu...")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error terjemahan batch {i}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error proses terjemahan: {str(e)}")
    finally:
        save_cache_to_file()


def process_single_row(text: str) -> List[Union[str, List[str]]]:
    """Memproses satu baris teks"""
    try:
        if pd.isna(text) or not isinstance(text, str):
            return ["", "", [], [], [], [], ""]

        # Langkah 1: Pembersihan teks
        cleaned = clean_text(text)
        if not cleaned:
            return ["", "", [], [], [], [], ""]

        # Langkah 2: Case folding
        lower = cleaned.lower()

        # Langkah 3: Tokenisasi
        tokens = word_tokenize(lower)
        if not tokens:
            return [cleaned, lower, [], [], [], [], ""]

        # Identifikasi kata non-Indonesia
        foreign_words = []
        for w in tokens:
            try:
                if len(w) <= 2:
                    continue
                if detect(w) != "id":
                    foreign_words.append(w)
            except LangDetectException:
                continue

        # Terjemahkan kata asing
        if foreign_words:
            translate_batch(foreign_words)

        # Gabungkan hasil terjemahan
        result_tokens = []
        for w in tokens:
            if len(w) <= 2:
                continue
            if w in translation_cache:
                result_tokens.extend(word_tokenize(translation_cache[w]))
            else:
                result_tokens.append(w)

        # Langkah 4: Stopword removal
        no_stopwords = remove_stopwords(result_tokens)

        # Langkah 5: Normalisasi
        normalized = normalize_text(no_stopwords)

        # Langkah 6: Stemming
        stemmed = stem_text(normalized)

        # Hasil akhir
        final_result = " ".join(stemmed)

        logger.debug(f"Diproses: {text[:50]}... → {final_result[:50]}...")
        return [cleaned, lower, tokens, no_stopwords, normalized, stemmed, final_result]

    except Exception as e:
        logger.error(f"Error memproses baris: {str(e)}")
        return ["", "", [], [], [], [], ""]


@prepro_bp.route("/preprocess", methods=["POST"])
def preprocess():
    """Endpoint utama untuk preprocessing teks bahasa Indonesia"""
    start_time = time.time()

    try:
        # 1. Validasi dasar file upload
        if "file" not in request.files:
            logger.error("Request tidak mengandung file")
            return jsonify({"error": "Harap unggah file CSV"}), 400

        file = request.files["file"]

        # 2. Validasi filename
        if file.filename is None:
            logger.error("Nama file tidak valid (None)")
            return jsonify({"error": "Nama file tidak valid"}), 400

        if not isinstance(file.filename, str):
            logger.error(f"Tipe nama file tidak valid: {type(file.filename)}")
            return jsonify({"error": "Format nama file tidak valid"}), 400

        if not file.filename.strip():
            logger.error("Nama file kosong")
            return jsonify({"error": "Nama file tidak boleh kosong"}), 400

        # 3. Validasi ekstensi file
        if not file.filename.strip().lower().endswith(".csv"):
            logger.error(f"Format file tidak didukung: {file.filename}")
            return jsonify({"error": "Hanya file CSV (.csv) yang diperbolehkan"}), 400

        # 4. Validasi ukuran file
        MAX_SIZE = 10 * 1024 * 1024  # 10MB
        if hasattr(file, "content_length") and file.content_length > MAX_SIZE:
            logger.error(f"File terlalu besar: {file.content_length} bytes")
            return jsonify({"error": "Ukuran file terlalu besar (maks 10MB)"}), 400

        # 5. Pastikan file memiliki stream
        if not hasattr(file, "stream"):
            logger.error("File tidak memiliki stream yang valid")
            return jsonify({"error": "Format file tidak valid"}), 400

        # 6. Proses file CSV
        try:
            with file.stream as f:
                chunks = pd.read_csv(f, chunksize=CHUNK_SIZE)
                processed_chunks = []
                total_rows = 0

                with ThreadPool(4) as pool:
                    for chunk_num, chunk in enumerate(chunks, 1):
                        logger.info(f"Memproses chunk {chunk_num} ({len(chunk)} baris)")

                        # 7. Validasi kolom 'Terjemahan'
                        if "Terjemahan" not in chunk.columns:
                            logger.error("Kolom 'Terjemahan' tidak ditemukan")
                            return jsonify(
                                {"error": "File harus mengandung kolom 'Terjemahan'"}
                            ), 400

                        chunk["Terjemahan"] = chunk["Terjemahan"].fillna("")
                        total_rows += len(chunk)

                        # 8. Proses setiap baris
                        results = pool.map(
                            process_single_row, chunk["Terjemahan"].tolist()
                        )

                        # 9. Gabungkan hasil
                        result_df = pd.DataFrame(
                            results,
                            columns=[
                                "Teks_Bersih",
                                "Case_Folding",
                                "Tokenisasi",
                                "Tanpa_Stopword",
                                "Normalisasi",
                                "Stemming",
                                "Hasil_Akhir",
                            ],
                        )

                        chunk = pd.concat(
                            [chunk.reset_index(drop=True), result_df], axis=1
                        )
                        chunk["Status"] = chunk.get("Status", "")
                        processed_chunks.append(chunk)

                # 10. Gabungkan semua chunk
                if not processed_chunks:
                    logger.error("Tidak ada data yang diproses")
                    return jsonify(
                        {"error": "File tidak mengandung data yang valid"}
                    ), 400

                final_df = pd.concat(processed_chunks, ignore_index=True)

                # 11. Simpan hasil
                export_df = final_df.copy()
                list_columns = [
                    "Tokenisasi",
                    "Tanpa_Stopword",
                    "Normalisasi",
                    "Stemming",
                ]

                for col in list_columns:
                    export_df[col] = export_df[col].apply(
                        lambda x: json.dumps(x, ensure_ascii=False)
                        if isinstance(x, list)
                        else "[]"
                    )

                os.makedirs(os.path.dirname(PREPRO_CSV_PATH), exist_ok=True)
                export_df.to_csv(
                    PREPRO_CSV_PATH,
                    index=False,
                    quoting=csv.QUOTE_ALL,
                    encoding="utf-8-sig",
                    lineterminator="\n",
                )

                # 12. Hitung statistik
                processing_time = time.time() - start_time
                rows_per_sec = (
                    total_rows / processing_time if processing_time > 0 else 0
                )

                logger.info(
                    f"Berhasil memproses {total_rows} baris "
                    f"dalam {processing_time:.2f} detik "
                    f"({rows_per_sec:.2f} baris/detik)"
                )

                return jsonify(
                    {
                        "status": "sukses",
                        "pesan": "Preprocessing berhasil",
                        "data": {
                            "total_baris": total_rows,
                            "waktu_proses": f"{processing_time:.2f} detik",
                            "kecepatan": f"{rows_per_sec:.2f} baris/detik",
                        },
                        "contoh_data": final_df.head(3).to_dict(orient="records"),
                    }
                )

        except pd.errors.EmptyDataError:
            logger.error("File CSV kosong")
            return jsonify({"error": "File CSV kosong"}), 400

        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV: {str(e)}")
            return jsonify({"error": "Format CSV tidak valid"}), 400

        except Exception as e:
            logger.error(f"Error memproses CSV: {str(e)}")
            return jsonify({"error": "Gagal memproses file CSV"}), 500

    except Exception as e:
        logger.error(f"Error tidak terduga: {str(e)}")
        return jsonify({"error": "Terjadi kesalahan server"}), 500

@prepro_bp.route("/save_csv", methods=["POST"])
def save_csv():
    """Endpoint untuk menyimpan data yang telah diproses"""
    try:
        data = request.json
        if not data:
            logger.error("Tidak ada data untuk disimpan")
            return jsonify({"error": "Tidak ada data untuk disimpan"}), 400

        df = pd.DataFrame(data)

        required_columns = [
            "SteamID",
            "Clean_Text",
            "Case_Folded",
            "Tokens",
            "No_Stopwords",
            "Normalized",
            "Stemmed",
            "Final_Result",
            "Status",
        ]

        for col in required_columns:
            if col not in df.columns:
                df[col] = ""

        df = df.reindex(columns=required_columns, fill_value="")

        list_columns = ["Tokens", "No_Stopwords", "Normalized", "Stemmed"]
        for col in list_columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False)
                if isinstance(x, list)
                else "[]"
            )

        for col in df.columns:
            if col not in list_columns:
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

        save_cache_to_file()

        logger.info("Data berhasil disimpan ke CSV")
        return send_file(
            PREPRO_CSV_PATH,
            mimetype="text/csv",
            as_attachment=True,
            download_name="processed_data.csv",
        )

    except Exception as e:
        logger.error(f"Gagal menyimpan CSV: {str(e)}")
        return jsonify({"error": f"Gagal menyimpan: {str(e)}"}), 500


@prepro_bp.route("/download")
def download_processed():
    """Endpoint untuk mengunduh hasil preprocessing"""
    if os.path.exists(PREPRO_CSV_PATH):
        logger.info("Mengirim file hasil preprocessing")
        return send_file(PREPRO_CSV_PATH, as_attachment=True)

    logger.error("File hasil tidak ditemukan")
    return jsonify({"error": "File hasil preprocessing tidak ditemukan"}), 404


@prepro_bp.route("/")
def main_view():
    """Endpoint tampilan utama"""
    return render_template("preprosessing.html", page_name="prepro"), 200


def register_template_filters(app):
    """Mendaftarkan filter template"""
    app.jinja_env.filters["sanitize"] = escape
    logger.info("Filter template berhasil didaftarkan")

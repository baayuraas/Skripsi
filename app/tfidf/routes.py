from flask import Blueprint, request, render_template, send_file, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import os
import pickle
import traceback
import re

tfidf_bp = Blueprint(
    "tfidf",
    __name__,
    url_prefix="/tfidf",
    template_folder="templates",
    static_folder="static",
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "tfidf")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Output paths
TEMP_OUTPUT = os.path.join(UPLOAD_FOLDER, "tfidf_output.csv")
TRAIN_CSV = os.path.join(UPLOAD_FOLDER, "tfidf_train.csv")
TEST_CSV = os.path.join(UPLOAD_FOLDER, "tfidf_test.csv")
TRAIN_MODEL = os.path.join(UPLOAD_FOLDER, "tfidf_train.pkl")
TEST_MODEL = os.path.join(UPLOAD_FOLDER, "tfidf_test.pkl")


def preprocess_text(text):
    """Preprocessing teks"""
    if not isinstance(text, str):
        return ""

    # Case folding
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_optimal_parameters(n_docs):
    """Parameter optimal berdasarkan ukuran dataset"""
    if n_docs < 300:
        # Small dataset
        return {"min_df": 1, "max_df": 0.85, "max_features": 800}
    elif n_docs <= 1000:
        # Medium dataset - optimal untuk 500-1000 dokumen
        return {"min_df": 1, "max_df": 0.8, "max_features": 1200}
    else:
        # Large dataset
        return {
            "min_df": max(1, int(0.001 * n_docs)),  # Minimal 1
            "max_df": 0.75,
            "max_features": 2000,
        }


@tfidf_bp.route("/")
def index():
    return render_template("tfidf.html", page_name="tfidf")


@tfidf_bp.route("/process", methods=["GET", "POST"])
def process_file():
    # Handle GET request - Load existing data
    if request.method == "GET":
        if not os.path.exists(TEMP_OUTPUT):
            return jsonify({"error": "Tidak ada data tersimpan"}), 404

        try:
            df = pd.read_csv(TEMP_OUTPUT)
            numeric_columns = [col for col in df.columns if col != "Status"]
            summary = {"total_documents": len(df), "total_terms": len(numeric_columns)}
            return jsonify({"data": df.to_dict(orient="records"), "summary": summary})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"Gagal memuat data existing: {str(e)}"}), 500

    # Handle POST request - Process new file
    file = request.files.get("file")
    if not file or not file.filename or not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "File tidak valid. Harus berformat .csv"}), 400

    try:
        df = pd.read_csv(file.stream)
        if "Hasil" not in df.columns or "Status" not in df.columns:
            return jsonify(
                {"error": "Kolom 'Hasil' atau 'Status' tidak ditemukan"}
            ), 400

        df["Hasil"] = df["Hasil"].fillna("").astype(str)
        df["Status"] = df["Status"].fillna("")

        if df["Hasil"].str.strip().eq("").all():
            return jsonify({"error": "Semua nilai pada kolom 'Hasil' kosong"}), 400

        # Preprocessing konsisten
        df["Hasil"] = df["Hasil"].apply(preprocess_text)

        # Split data 70:30
        train_df, test_df = train_test_split(
            df, test_size=0.3, random_state=42, stratify=df["Status"]
        )

        # Dapatkan parameter optimal
        n_docs = len(df)
        params = get_optimal_parameters(n_docs)

        # Train vectorizer dengan parameter optimal (min_df=1)
        tfidf_vectorizer = TfidfVectorizer(
            min_df=params["min_df"],  # Sekarang 1 untuk semua ukuran dataset
            max_df=params["max_df"],
            max_features=params["max_features"],
            smooth_idf=True,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )

        tfidf_train_matrix = tfidf_vectorizer.fit_transform(train_df["Hasil"])
        tfidf_test_matrix = tfidf_vectorizer.transform(test_df["Hasil"])
        tfidf_full_matrix = tfidf_vectorizer.transform(df["Hasil"])

        # Save models
        with open(TRAIN_MODEL, "wb") as f:
            pickle.dump(tfidf_vectorizer, f)
        with open(TEST_MODEL, "wb") as f:
            pickle.dump(tfidf_vectorizer, f)

        terms = tfidf_vectorizer.get_feature_names_out()

        def build_tfidf_df(matrix, src_df):
            dense_matrix = (
                matrix.toarray()
                if isinstance(matrix, csr_matrix)
                else np.asarray(matrix.todense())
            )
            tfidf_df = pd.DataFrame(dense_matrix, columns=terms)
            tfidf_df["Status"] = src_df["Status"].values
            ordered_columns = [c for c in tfidf_df.columns if c != "Status"] + [
                "Status"
            ]
            return tfidf_df[ordered_columns].fillna(0)

        tfidf_train_df = build_tfidf_df(tfidf_train_matrix, train_df)
        tfidf_test_df = build_tfidf_df(tfidf_test_matrix, test_df)
        tfidf_full_df = build_tfidf_df(tfidf_full_matrix, df)

        # Save CSVs
        tfidf_train_df.to_csv(TRAIN_CSV, index=False)
        tfidf_test_df.to_csv(TEST_CSV, index=False)
        tfidf_full_df.to_csv(TEMP_OUTPUT, index=False)

        # Siapkan response data
        response_data = tfidf_full_df.to_dict(orient="records")

        # Summary information
        summary = {"total_documents": len(tfidf_full_df), "total_terms": len(terms)}

        return jsonify({"data": response_data, "summary": summary})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses TF-IDF: {str(e)}"}), 500


@tfidf_bp.route("/download/<tipe>")
def download_by_type(tipe):
    mapping = {
        "train_csv": TRAIN_CSV,
        "test_csv": TEST_CSV,
        "train_model": TRAIN_MODEL,
        "test_model": TEST_MODEL,
        "full_csv": TEMP_OUTPUT,
    }
    path = mapping.get(tipe)
    if not path or not os.path.exists(path):
        return jsonify({"error": f"File {tipe} tidak ditemukan"}), 404
    return send_file(path, as_attachment=True)


@tfidf_bp.route("/clear", methods=["POST"])
def clear_data():
    removed = []
    files_to_remove = [TEMP_OUTPUT, TRAIN_CSV, TEST_CSV, TRAIN_MODEL, TEST_MODEL]

    for f in files_to_remove:
        if os.path.exists(f):
            try:
                os.remove(f)
                removed.append(f)
            except Exception as e:
                return jsonify({"error": f"Gagal menghapus file {f}: {str(e)}"}), 500

    if removed:
        return jsonify({"message": "Data TF-IDF telah dihapus"}), 200
    return jsonify({"error": "Tidak ada data TF-IDF yang perlu dihapus"}), 400

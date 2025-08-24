from flask import Blueprint, request, render_template, send_file, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import os
import pickle
import traceback

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


@tfidf_bp.route("/")
def index():
    return render_template("tfidf.html", page_name="tfidf")


@tfidf_bp.route("/process", methods=["POST"])
def process_file():
    file = request.files.get("file")
    if not file or not file.filename or not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "File tidak valid. Harus berformat .csv"}), 400

    try:
        df = pd.read_csv(file.stream)
        if "Hasil" not in df.columns or "Status" not in df.columns:
            return jsonify({"error": "Kolom 'Hasil' atau 'Status' tidak ditemukan"}), 400

        df["Hasil"] = df["Hasil"].fillna("").astype(str)
        df["Status"] = df["Status"].fillna("")

        if df["Hasil"].str.strip().eq("").all():
            return jsonify({"error": "Semua nilai pada kolom 'Hasil' kosong"}), 400

        # Split data 70:30
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["Status"])

        # Train vectorizer on train data
        tfidf_vectorizer = TfidfVectorizer(min_df=2, dtype=np.float32)
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
            dense_matrix = matrix.toarray() if isinstance(matrix, csr_matrix) else np.asarray(matrix.todense())
            tfidf_df = pd.DataFrame(dense_matrix, columns=terms)
            tfidf_df["Status"] = src_df["Status"].values
            ordered_columns = [c for c in tfidf_df.columns if c != "Status"] + ["Status"]
            return tfidf_df[ordered_columns].fillna(0)

        tfidf_train_df = build_tfidf_df(tfidf_train_matrix, train_df)
        tfidf_test_df = build_tfidf_df(tfidf_test_matrix, test_df)
        tfidf_full_df = build_tfidf_df(tfidf_full_matrix, df)

        # Save CSVs
        tfidf_train_df.to_csv(TRAIN_CSV, index=False)
        tfidf_test_df.to_csv(TEST_CSV, index=False)
        tfidf_full_df.to_csv(TEMP_OUTPUT, index=False)

        return jsonify({"data": tfidf_full_df.to_dict(orient="records")})

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
    for f in [TEMP_OUTPUT, TRAIN_CSV, TEST_CSV, TRAIN_MODEL, TEST_MODEL]:
        if os.path.exists(f):
            os.remove(f)
            removed.append(f)
    if removed:
        return jsonify({"message": "Data TF-IDF telah dihapus"}), 200
    return jsonify({"error": "Tidak ada data TF-IDF yang perlu dihapus"}), 400

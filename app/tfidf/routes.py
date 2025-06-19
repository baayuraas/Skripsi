from flask import Blueprint, request, render_template, send_file, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from scipy.sparse import csr_matrix

tfidf_bp = Blueprint(
    "tfidf",
    __name__,
    url_prefix="/tfidf",
    template_folder="templates",
    static_folder="static",
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "tfidf")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

TEMP_OUTPUT = os.path.join(UPLOAD_FOLDER, "tfidf_output.csv")


@tfidf_bp.route("/")
def index():
    return render_template("tfidf.html")


@tfidf_bp.route("/process", methods=["POST"])
def process_file():
    file = request.files.get("file")

    if not file or not file.filename or not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "File tidak valid. Harus berformat .csv"}), 400

    try:
        df = pd.read_csv(file.stream)
        print("CSV dimuat, kolom tersedia:", df.columns.tolist())

        if "Hasil" not in df.columns:
            return jsonify(
                {"error": "Kolom 'Hasil' tidak ditemukan dalam file CSV"}
            ), 400
        if "Status" not in df.columns:
            return jsonify(
                {"error": "Kolom 'Status' tidak ditemukan dalam file CSV"}
            ), 400

        df["Hasil"] = df["Hasil"].fillna("").astype(str)
        df["Status"] = df["Status"].fillna("")

        if df["Hasil"].str.strip().eq("").all():
            return jsonify({"error": "Semua nilai pada kolom 'Hasil' kosong"}), 400

        tfidf_vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(df["Hasil"])
        except ValueError as ve:
            return jsonify({"error": f"Gagal membentuk TF-IDF: {str(ve)}"}), 400

        terms = tfidf_vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(csr_matrix(tfidf_matrix).toarray(), columns=terms)
        tfidf_df["Status"] = df["Status"]

        ordered_columns = [col for col in tfidf_df.columns if col != "Status"] + [
            "Status"
        ]
        tfidf_df = tfidf_df[ordered_columns].fillna(0)

        os.makedirs(os.path.dirname(TEMP_OUTPUT), exist_ok=True)
        tfidf_df.to_csv(TEMP_OUTPUT, index=False)

        return jsonify({"data": tfidf_df.to_dict(orient="records")})
    except Exception as e:
        import traceback

        traceback.print_exc()  # Tampilkan detail error di terminal Flask
        return jsonify({"error": f"Gagal memproses TF-IDF: {str(e)}"}), 500


@tfidf_bp.route("/download")
def download_file():
    if os.path.exists(TEMP_OUTPUT):
        return send_file(TEMP_OUTPUT, as_attachment=True)
    return jsonify({"error": "File hasil TF-IDF tidak ditemukan"}), 404


@tfidf_bp.route("/clear", methods=["POST"])
def clear_data():
    if os.path.exists(TEMP_OUTPUT):
        os.remove(TEMP_OUTPUT)
        return jsonify({"message": "Data TF-IDF telah dihapus"}), 200
    return jsonify({"error": "Tidak ada data TF-IDF yang perlu dihapus"}), 400

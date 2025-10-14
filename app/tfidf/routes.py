from flask import Blueprint, request, render_template, send_file, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, hstack
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

# Output paths untuk CSV Limited (Display)
TEMP_OUTPUT = os.path.join(UPLOAD_FOLDER, "tfidf_output.csv")
TRAIN_CSV = os.path.join(UPLOAD_FOLDER, "tfidf_train.csv")
TEST_CSV = os.path.join(UPLOAD_FOLDER, "tfidf_test.csv")

# Output paths untuk CSV Unlimited (Semua Term)
UNLIMITED_CSV = os.path.join(UPLOAD_FOLDER, "tfidf_unlimited_output.csv")
UNLIMITED_TRAIN_CSV = os.path.join(UPLOAD_FOLDER, "tfidf_unlimited_train.csv")
UNLIMITED_TEST_CSV = os.path.join(UPLOAD_FOLDER, "tfidf_unlimited_test.csv")

# Output paths untuk Model
TRAIN_MODEL = os.path.join(UPLOAD_FOLDER, "tfidf_train_model.pkl")
TEST_MODEL = os.path.join(UPLOAD_FOLDER, "tfidf_test_model.pkl")
UNLIMITED_MODEL = os.path.join(UPLOAD_FOLDER, "tfidf_unlimited_model.pkl")
METADATA_FILE = os.path.join(UPLOAD_FOLDER, "tfidf_metadata.pkl")


def preprocess_text(text):
    """Preprocessing teks"""
    if not isinstance(text, str):
        return ""

    # Case folding
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_display_parameters(n_docs):
    """Parameter untuk pembatasan display frontend"""
    if n_docs < 300:
        return {"max_features": 800}
    elif n_docs <= 1000:
        return {"max_features": 1200}
    else:
        return {"max_features": 2000}


def get_top_terms_from_matrix(matrix, feature_names, n_top_terms=1000):
    """
    Pilih term-term terbaik dari matriks unlimited berdasarkan skor TF-IDF
    """
    # Hitung importance score untuk setiap term (rata-rata TF-IDF)
    term_scores = np.mean(matrix.toarray(), axis=0)

    # Dapatkan indeks term dengan skor tertinggi
    top_indices = np.argsort(term_scores)[::-1][:n_top_terms]

    # Return term names dan indices
    top_terms = [feature_names[i] for i in top_indices]
    return top_terms, top_indices


def extract_columns_sparse(matrix, column_indices):
    """Extract specific columns from sparse matrix safely"""
    if matrix.shape[1] == 0 or len(column_indices) == 0:
        return csr_matrix((matrix.shape[0], len(column_indices)))

    # Konversi ke CSR format untuk indexing yang lebih efisien
    matrix_csr = matrix.tocsr()

    # Filter hanya indices yang valid
    valid_indices = [idx for idx in column_indices if idx < matrix_csr.shape[1]]

    if len(valid_indices) == 0:
        return csr_matrix((matrix_csr.shape[0], len(column_indices)))

    # Extract columns menggunakan slicing
    result = matrix_csr[:, valid_indices]

    # Jika ada columns yang missing, kita perlu menambahkan columns kosong
    if len(valid_indices) < len(column_indices):
        missing_cols = len(column_indices) - len(valid_indices)
        empty_cols = csr_matrix((matrix_csr.shape[0], missing_cols))
        result = hstack([result, empty_cols])

    return result


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

            # Coba baca metadata
            total_terms_unlimited = 0
            if os.path.exists(METADATA_FILE):
                try:
                    with open(METADATA_FILE, "rb") as f:
                        metadata = pickle.load(f)
                        total_terms_unlimited = metadata.get("total_terms_unlimited", 0)
                except (FileNotFoundError, pickle.PickleError, EOFError, KeyError) as e:
                    print(f"Warning: Gagal memuat metadata: {str(e)}")
                    total_terms_unlimited = 0

            summary = {
                "total_documents": len(df),
                "total_terms": len(numeric_columns),
                "total_terms_unlimited": total_terms_unlimited,
            }
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

        # Split data
        train_df, test_df = train_test_split(
            df, test_size=0.5, random_state=42, stratify=df["Status"]
        )

        # Dapatkan parameter untuk display
        n_docs = len(df)
        display_params = get_display_parameters(n_docs)
        n_display_terms = display_params["max_features"]

        # **STEP 1: HITUNG TF-IDF DENGAN SEMUA TERM (UNLIMITED) - UNTUK SEMUA PERHITUNGAN**
        print("🔄 Menghitung TF-IDF dengan semua term (unlimited)...")
        tfidf_unlimited = TfidfVectorizer(
            min_df=1,  # Minimal 1 dokumen
            max_df=1.0,  # Maksimal 100% dokumen
            max_features=None,  # TANPA BATASAN - gunakan semua term
            smooth_idf=True,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )

        # Fit dan transform dengan SEMUA data menggunakan unlimited vectorizer
        tfidf_unlimited_matrix = tfidf_unlimited.fit_transform(df["Hasil"])
        all_feature_names = tfidf_unlimited.get_feature_names_out()
        total_terms_unlimited = len(all_feature_names)
        print(f"✅ Total terms tanpa batasan: {total_terms_unlimited}")

        # **STEP 2: BUAT MODEL TERPISAH UNTUK TRAIN DAN TEST**
        print("🔧 Membuat model terpisah untuk train dan test...")

        # Model untuk data train (hanya di-fit pada data train)
        tfidf_train_model = TfidfVectorizer(
            min_df=1,
            max_df=1.0,
            max_features=None,
            smooth_idf=True,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        tfidf_train_matrix = tfidf_train_model.fit_transform(train_df["Hasil"])

        # Model untuk data test (hanya di-fit pada data test)
        tfidf_test_model = TfidfVectorizer(
            min_df=1,
            max_df=1.0,
            max_features=None,
            smooth_idf=True,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        tfidf_test_matrix = tfidf_test_model.fit_transform(test_df["Hasil"])

        # **STEP 3: SIMPAN SEMUA MODEL**
        print("💾 Menyimpan semua model...")
        with open(UNLIMITED_MODEL, "wb") as f:
            pickle.dump(tfidf_unlimited, f)
        with open(TRAIN_MODEL, "wb") as f:
            pickle.dump(tfidf_train_model, f)
        with open(TEST_MODEL, "wb") as f:
            pickle.dump(tfidf_test_model, f)

        # **STEP 4: BUAT CSV UNLIMITED (SEMUA TERM)**
        print("📊 Membuat CSV unlimited...")

        def build_unlimited_df(matrix, feature_names, src_df):
            """Build DataFrame dengan SEMUA term"""
            dense_matrix = (
                matrix.toarray()
                if isinstance(matrix, csr_matrix)
                else np.asarray(matrix.todense())
            )
            tfidf_df = pd.DataFrame(dense_matrix, columns=feature_names)
            tfidf_df["Status"] = src_df["Status"].values
            ordered_columns = [c for c in tfidf_df.columns if c != "Status"] + [
                "Status"
            ]
            return tfidf_df[ordered_columns].fillna(0)

        # Build DataFrame unlimited
        tfidf_unlimited_df = build_unlimited_df(
            tfidf_unlimited_matrix, all_feature_names, df
        )
        tfidf_unlimited_train_df = build_unlimited_df(
            tfidf_train_matrix, tfidf_train_model.get_feature_names_out(), train_df
        )
        tfidf_unlimited_test_df = build_unlimited_df(
            tfidf_test_matrix, tfidf_test_model.get_feature_names_out(), test_df
        )

        # Save unlimited CSVs
        tfidf_unlimited_df.to_csv(UNLIMITED_CSV, index=False)
        tfidf_unlimited_train_df.to_csv(UNLIMITED_TRAIN_CSV, index=False)
        tfidf_unlimited_test_df.to_csv(UNLIMITED_TEST_CSV, index=False)

        # **STEP 5: PILIH TERM TERBAIK UNTUK DISPLAY FRONTEND (LIMITED)**
        print("🔍 Memilih term terbaik untuk display frontend...")
        top_terms, top_indices = get_top_terms_from_matrix(
            tfidf_unlimited_matrix, all_feature_names, n_top_terms=n_display_terms
        )

        print(
            f"✅ Term terpilih untuk display: {len(top_terms)} dari {total_terms_unlimited}"
        )

        # **STEP 6: BUAT MATRIKS DISPLAY HANYA DENGAN TERM TERPILIH (LIMITED)**
        print("🔧 Membuat matriks display dengan term terpilih...")

        # Apply extract columns ke semua matriks
        tfidf_display_matrix = extract_columns_sparse(
            tfidf_unlimited_matrix, top_indices
        )
        tfidf_train_display = extract_columns_sparse(tfidf_train_matrix, top_indices)
        tfidf_test_display = extract_columns_sparse(tfidf_test_matrix, top_indices)

        print("✅ Matriks display berhasil dibuat")
        print(f"   - Display shape: {tfidf_display_matrix.shape}")
        print(f"   - Train shape: {tfidf_train_display.shape}")
        print(f"   - Test shape: {tfidf_test_display.shape}")

        # Save metadata
        metadata = {
            "total_terms_unlimited": total_terms_unlimited,
            "display_features": len(top_terms),
            "top_terms": top_terms,
        }
        try:
            with open(METADATA_FILE, "wb") as f:
                pickle.dump(metadata, f)
        except Exception as e:
            print(f"Warning: Gagal menyimpan metadata: {str(e)}")

        # **STEP 7: BUAT DATAFRAME UNTUK DISPLAY FRONTEND (LIMITED)**
        def build_display_df(matrix, feature_names, src_df):
            """Build DataFrame hanya dengan term-term terpilih untuk display"""
            # Konversi ke dense matrix hanya jika diperlukan
            if isinstance(matrix, csr_matrix):
                dense_matrix = matrix.toarray()
            else:
                dense_matrix = np.asarray(matrix.todense())

            # Pastikan jumlah feature names sesuai dengan columns
            n_cols = dense_matrix.shape[1]
            if len(feature_names) > n_cols:
                # Jika feature names lebih banyak, ambil yang diperlukan saja
                display_features = feature_names[:n_cols]
            elif len(feature_names) < n_cols:
                # Jika feature names kurang, tambahkan nama default
                display_features = list(feature_names) + [
                    f"term_{i}" for i in range(len(feature_names), n_cols)
                ]
            else:
                display_features = feature_names

            tfidf_df = pd.DataFrame(dense_matrix, columns=display_features)
            tfidf_df["Status"] = src_df["Status"].values
            ordered_columns = [c for c in tfidf_df.columns if c != "Status"] + [
                "Status"
            ]
            return tfidf_df[ordered_columns].fillna(0)

        # Build DataFrame untuk display (LIMITED)
        tfidf_display_df = build_display_df(tfidf_display_matrix, top_terms, df)
        tfidf_train_df = build_display_df(tfidf_train_display, top_terms, train_df)
        tfidf_test_df = build_display_df(tfidf_test_display, top_terms, test_df)

        # Save CSVs (LIMITED - untuk display)
        tfidf_train_df.to_csv(TRAIN_CSV, index=False)
        tfidf_test_df.to_csv(TEST_CSV, index=False)
        tfidf_display_df.to_csv(TEMP_OUTPUT, index=False)

        # Siapkan response data untuk frontend (LIMITED)
        response_data = tfidf_display_df.to_dict(orient="records")

        # Summary information
        summary = {
            "total_documents": len(tfidf_display_df),
            "total_terms": len(top_terms),  # Jumlah term yang ditampilkan (limited)
            "total_terms_unlimited": total_terms_unlimited,  # Jumlah term sebenarnya (unlimited)
        }

        print("🎉 Proses TF-IDF selesai!")
        print(f"   - Dokumen: {summary['total_documents']}")
        print(f"   - Term unlimited (perhitungan): {summary['total_terms_unlimited']}")
        print(f"   - Term display (frontend): {summary['total_terms']}")
        print("   - File yang disimpan:")
        print("     • CSV Unlimited: Full, Train, Test")
        print("     • CSV Limited: Full, Train, Test")
        print("     • Model: Unlimited, Train, Test")

        return jsonify({"data": response_data, "summary": summary})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses TF-IDF: {str(e)}"}), 500


@tfidf_bp.route("/download/<tipe>")
def download_by_type(tipe):
    mapping = {
        # CSV Limited (Display)
        "train_csv": TRAIN_CSV,
        "test_csv": TEST_CSV,
        "full_csv": TEMP_OUTPUT,
        # CSV Unlimited (Semua Term)
        "unlimited_train_csv": UNLIMITED_TRAIN_CSV,
        "unlimited_test_csv": UNLIMITED_TEST_CSV,
        "unlimited_csv": UNLIMITED_CSV,
        # Model
        "train_model": TRAIN_MODEL,
        "test_model": TEST_MODEL,
        "unlimited_model": UNLIMITED_MODEL,
    }
    path = mapping.get(tipe)
    if not path or not os.path.exists(path):
        return jsonify({"error": f"File {tipe} tidak ditemukan"}), 404
    return send_file(path, as_attachment=True)


@tfidf_bp.route("/clear", methods=["POST"])
def clear_data():
    removed = []
    files_to_remove = [
        # CSV Limited
        TEMP_OUTPUT,
        TRAIN_CSV,
        TEST_CSV,
        # CSV Unlimited
        UNLIMITED_CSV,
        UNLIMITED_TRAIN_CSV,
        UNLIMITED_TEST_CSV,
        # Model
        TRAIN_MODEL,
        TEST_MODEL,
        UNLIMITED_MODEL,
        # Metadata
        METADATA_FILE,
    ]

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

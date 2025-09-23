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
from collections import Counter

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
    """Preprocessing konsisten untuk teks"""
    if not isinstance(text, str):
        return ""
    
    # Case folding
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def analyze_term_distribution(texts, specific_terms=None):
    """Analisis distribusi term tertentu"""
    if specific_terms is None:
        specific_terms = ['main', 'dota']  # Terms yang ingin dianalisis
    
    results = {}
    all_texts = [str(text) for text in texts]
    
    for term in specific_terms:
        # Hitung di berapa dokumen term muncul
        doc_count = sum(1 for text in all_texts if re.search(rf'\b{re.escape(term)}\b', text.lower()))
        
        # Hitung total frekuensi
        all_words = ' '.join(all_texts).lower().split()
        total_count = sum(1 for word in all_words if word == term)
        
        results[term] = {
            'document_frequency': doc_count,
            'total_frequency': total_count
        }
    
    return results

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

        # Preprocessing konsisten
        df["Hasil"] = df["Hasil"].apply(preprocess_text)

        # Debug: Analisis distribusi term sebelum TF-IDF
        print("=== ANALISIS DISTRIBUSI TERM ===")
        term_analysis = analyze_term_distribution(df["Hasil"])
        for term, stats in term_analysis.items():
            print(f"Term '{term}': muncul di {stats['document_frequency']} dokumen, total {stats['total_frequency']} kali")

        # Split data 70:30
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["Status"])

        # Debug: Analisis distribusi di data training
        train_analysis = analyze_term_distribution(train_df["Hasil"])
        print("=== DISTRIBUSI DI DATA TRAINING ===")
        for term, stats in train_analysis.items():
            print(f"Term '{term}': muncul di {stats['document_frequency']} dokumen training")

        # Train vectorizer dengan parameter yang lebih optimal
        tfidf_vectorizer = TfidfVectorizer(
            min_df=1,           # Term muncul minimal di 1 dokumen
            max_df=0.95,        # Abaikan term yang muncul di >95% dokumen (stopwords)
            token_pattern=r'(?u)\b\w+\b',  # Pattern untuk menangkap kata
            dtype=np.float32
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

        # Debug: Cek vocabulary
        print("=== VOCABULARY ANALYSIS ===")
        print(f"Total terms dalam vocabulary: {len(terms)}")
        for term in ['main', 'dota']:
            if term in terms:
                print(f"Term '{term}' ADA dalam vocabulary")
            else:
                print(f"Term '{term}' TIDAK ADA dalam vocabulary")

        def build_tfidf_df(matrix, src_df):
            dense_matrix = matrix.toarray() if isinstance(matrix, csr_matrix) else np.asarray(matrix.todense())
            tfidf_df = pd.DataFrame(dense_matrix, columns=terms)
            tfidf_df["Status"] = src_df["Status"].values
            tfidf_df["Original_Text"] = src_df["Hasil"].values  # Tambahkan teks asli untuk debugging
            ordered_columns = ["Original_Text"] + [c for c in tfidf_df.columns if c not in ["Original_Text", "Status"]] + ["Status"]
            return tfidf_df[ordered_columns].fillna(0)

        tfidf_train_df = build_tfidf_df(tfidf_train_matrix, train_df)
        tfidf_test_df = build_tfidf_df(tfidf_test_matrix, test_df)
        tfidf_full_df = build_tfidf_df(tfidf_full_matrix, df)

        # Debug: Analisis baris dengan nilai 0
        print("=== ANALISIS BARIS DENGAN NILAI 0 ===")
        # Hanya kolom numeric (exclude Status dan Original_Text)
        numeric_columns = [col for col in tfidf_full_df.columns if col not in ['Status', 'Original_Text']]
        zero_rows_mask = (tfidf_full_df[numeric_columns] == 0).all(axis=1)
        zero_rows_count = zero_rows_mask.sum()
        
        if zero_rows_count > 0:
            print(f"Ditemukan {zero_rows_count} baris dengan semua nilai TF-IDF = 0")
            zero_rows_indices = tfidf_full_df[zero_rows_mask].index.tolist()
            
            # Analisis beberapa baris contoh
            for i, idx in enumerate(zero_rows_indices[:3]):  # Limit to first 3 for debugging
                original_text = tfidf_full_df.loc[idx, 'Original_Text']
                print(f"Baris {idx}: '{original_text}'")
                
                # Cek term spesifik dalam teks
                words_in_text = set(re.findall(r'\b\w+\b', original_text.lower()))
                common_terms = words_in_text.intersection(['main', 'dota'])
                if common_terms:
                    print(f"  Term yang seharusnya ada: {common_terms}")
        
        # Debug: Cek nilai untuk term tertentu
        if 'main' in terms:
            main_col = tfidf_full_df['main']
            non_zero_main = (main_col > 0).sum()
            print(f"Term 'main' memiliki nilai non-zero di {non_zero_main} dokumen")
            if non_zero_main > 0:
                sample_non_zero = main_col[main_col > 0].head(3)
                for idx, val in sample_non_zero.items():
                    original_text = tfidf_full_df.loc[idx, 'Original_Text']
                    print(f"  Dokumen {idx}: nilai={val:.4f}, teks='{original_text}'")

        # Save CSVs
        tfidf_train_df.to_csv(TRAIN_CSV, index=False)
        tfidf_test_df.to_csv(TEST_CSV, index=False)
        
        # Untuk file full, hapus kolom Original_Text sebelum disimpan
        tfidf_full_df_without_text = tfidf_full_df.drop('Original_Text', axis=1)
        tfidf_full_df_without_text.to_csv(TEMP_OUTPUT, index=False)

        # Siapkan response data tanpa Original_Text untuk ditampilkan di UI
        response_data = tfidf_full_df_without_text.to_dict(orient="records")
        
        # Tambahkan summary information
        summary = {
            "total_documents": len(tfidf_full_df),
            "total_terms": len(terms),
            "zero_value_rows": int(zero_rows_count),
            "term_analysis": term_analysis
        }

        return jsonify({
            "data": response_data,
            "summary": summary
        })

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

@tfidf_bp.route("/load_existing")
def load_existing_data():
    if not os.path.exists(TEMP_OUTPUT):
        return jsonify({"error": "Tidak ada data tersimpan"}), 404
    
    try:
        df = pd.read_csv(TEMP_OUTPUT)
        
        # Tambahkan summary information
        numeric_columns = [col for col in df.columns if col != 'Status']
        zero_rows_count = (df[numeric_columns] == 0).all(axis=1).sum()
        
        summary = {
            "total_documents": len(df),
            "total_terms": len(numeric_columns),
            "zero_value_rows": int(zero_rows_count)
        }
        
        return jsonify({
            "data": df.to_dict(orient="records"),
            "summary": summary
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memuat data existing: {str(e)}"}), 500

@tfidf_bp.route("/debug_document/<int:doc_index>")
def debug_document(doc_index):
    """Endpoint khusus untuk debugging dokumen tertentu"""
    if not os.path.exists(TEMP_OUTPUT):
        return jsonify({"error": "Tidak ada data tersimpan"}), 404
    
    try:
        df = pd.read_csv(TEMP_OUTPUT)
        
        if doc_index >= len(df):
            return jsonify({"error": f"Index {doc_index} diluar range data"}), 400
        
        # Baca file original untuk mendapatkan teks asli
        original_file_path = os.path.join(UPLOAD_FOLDER, "original_data.csv")
        original_text = "Teks asli tidak tersedia"
        if os.path.exists(original_file_path):
            original_df = pd.read_csv(original_file_path)
            if doc_index < len(original_df) and "Hasil" in original_df.columns:
                original_text = original_df.iloc[doc_index]["Hasil"]
        
        doc_data = df.iloc[doc_index]
        numeric_columns = [col for col in df.columns if col != 'Status']
        
        # Hitung term dengan nilai non-zero
        non_zero_terms = []
        for col in numeric_columns:
            if doc_data[col] > 0:
                non_zero_terms.append({
                    "term": col,
                    "value": float(doc_data[col])
                })
        
        debug_info = {
            "document_index": doc_index,
            "original_text": original_text,
            "status": doc_data.get('Status', 'Unknown'),
            "total_terms": len(numeric_columns),
            "non_zero_terms_count": len(non_zero_terms),
            "non_zero_terms": non_zero_terms,
            "all_zero": len(non_zero_terms) == 0
        }
        
        return jsonify(debug_info)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal debugging dokumen: {str(e)}"}), 500
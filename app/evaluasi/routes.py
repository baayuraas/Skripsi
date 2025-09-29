import os
import numpy as np
import pandas as pd
import pickle
import traceback
from flask import Blueprint, request, jsonify, render_template, send_file
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras.models import load_model

evaluasi_bp = Blueprint(
    "evaluasi",
    __name__,
    url_prefix="/evaluasi",
    template_folder="templates",
    static_folder="static",
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "evaluasi")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

EVAL_PATH = os.path.join(UPLOAD_FOLDER, "data_eval.csv")
MODEL_PATH = os.path.join(BASE_DIR, "uploads", "perhitungan", "model_mlp_custom.keras")
LABEL_ENCODER_PATH = os.path.join(
    BASE_DIR, "uploads", "perhitungan", "label_encoder.pkl"
)
RESULTS_PATH = os.path.join(UPLOAD_FOLDER, "hasil_evaluasi.csv")
SAMPLED_DATA_PATH = os.path.join(UPLOAD_FOLDER, "data_sampled.csv")


@evaluasi_bp.route("/")
def index():
    return render_template("evaluasi.html", page_name="evaluasi")


@evaluasi_bp.route("/check_existing_data", methods=["GET"])
def check_existing_data():
    """Memeriksa apakah ada data evaluasi yang tersimpan"""
    if not os.path.exists(EVAL_PATH):
        return jsonify({"exists": False})

    try:
        # Coba baca file untuk memastikan valid
        df = pd.read_csv(EVAL_PATH)
        if df.empty:
            return jsonify({"exists": False})

        return jsonify({"exists": True, "row_count": len(df)})
    except Exception:
        return jsonify({"exists": False})


@evaluasi_bp.route("/sample_data", methods=["POST"])
def sample_data():
    """Mengambil sampel data berdasarkan input user"""
    try:
        data = request.get_json()
        total_samples = data.get("total_samples", 0)
        positive_samples = data.get("positive_samples", 0)
        negative_samples = data.get("negative_samples", 0)

        if not os.path.exists(EVAL_PATH):
            return jsonify({"error": "Tidak ada data evaluasi yang tersimpan"}), 400

        df = pd.read_csv(EVAL_PATH)

        # Validasi input
        if total_samples <= 0 or positive_samples < 0 or negative_samples < 0:
            return jsonify({"error": "Jumlah sampel harus positif"}), 400

        if positive_samples + negative_samples != total_samples:
            return jsonify(
                {"error": "Jumlah positif + negatif harus sama dengan total sampel"}
            ), 400

        # Cari kolom label (asumsi kolom terakhir adalah label)
        label_column = df.columns[-1]

        # Identifikasi nilai positif dan negatif
        unique_labels = df[label_column].unique()
        if len(unique_labels) != 2:
            return jsonify(
                {"error": "Data harus memiliki tepat 2 kelas (positif dan negatif)"}
            ), 400

        # Asumsikan label pertama adalah positif, kedua negatif
        positive_label = unique_labels[0]
        negative_label = unique_labels[1]

        # Pisahkan data berdasarkan label
        positive_data = df[df[label_column] == positive_label]
        negative_data = df[df[label_column] == negative_label]

        # Validasi jumlah data yang tersedia
        if len(positive_data) < positive_samples:
            return jsonify(
                {
                    "error": f"Data positif hanya {len(positive_data)}, tidak cukup untuk {positive_samples} sampel"
                }
            ), 400

        if len(negative_data) < negative_samples:
            return jsonify(
                {
                    "error": f"Data negatif hanya {len(negative_data)}, tidak cukup untuk {negative_samples} sampel"
                }
            ), 400

        # Ambil sampel secara acak
        sampled_positive = positive_data.sample(n=positive_samples, random_state=42)
        sampled_negative = negative_data.sample(n=negative_samples, random_state=42)

        # Gabungkan sampel
        sampled_df = pd.concat([sampled_positive, sampled_negative], ignore_index=True)

        # Acak urutan data
        sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Simpan data sampel
        sampled_df.to_csv(SAMPLED_DATA_PATH, index=False)

        return jsonify(
            {
                "message": f"Berhasil mengambil {total_samples} sampel ({positive_samples} positif, {negative_samples} negatif)",
                "sampled_data": sampled_df.to_dict(orient="records"),
                "total_sampled": len(sampled_df),
            }
        )

    except Exception as e:
        print(f"Error in sample_data: {str(e)}")
        return jsonify({"error": f"Gagal mengambil sampel: {str(e)}"}), 500


@evaluasi_bp.route("/evaluate_sampled", methods=["POST"])
def evaluate_sampled():
    """Melakukan evaluasi menggunakan data sampel"""
    if not os.path.exists(SAMPLED_DATA_PATH):
        return jsonify({"error": "Tidak ada data sampel yang tersimpan"}), 400

    return evaluate_data(SAMPLED_DATA_PATH)


@evaluasi_bp.route("/evaluate_existing", methods=["POST"])
def evaluate_existing():
    """Melakukan evaluasi menggunakan data yang sudah ada"""
    return evaluate_data(EVAL_PATH)


def evaluate_data(file_path):
    """Fungsi helper untuk evaluasi data"""
    missing_files = []
    if not os.path.exists(MODEL_PATH):
        missing_files.append(f"❌ Model tidak ditemukan di: {MODEL_PATH}")
    if not os.path.exists(LABEL_ENCODER_PATH):
        missing_files.append(
            f"❌ Label Encoder tidak ditemukan di: {LABEL_ENCODER_PATH}"
        )

    if missing_files:
        return jsonify({"error": "Gagal evaluasi:\n" + "\n".join(missing_files)}), 400

    try:
        df = pd.read_csv(file_path)

        df_features = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
        valid_mask = df_features.notna().all(axis=1)
        df_clean = df[valid_mask].copy()
        skipped_rows = len(df) - len(df_clean)

        if df_clean.empty:
            return jsonify({"error": "Tidak ada baris data valid di CSV."}), 400

        X = df_clean.iloc[:, :-1].to_numpy()
        y_true_labels = df_clean.iloc[:, -1].to_numpy()

        with open(LABEL_ENCODER_PATH, "rb") as f:
            le: LabelEncoder = pickle.load(f)

        try:
            y_true = le.transform(y_true_labels)
        except ValueError as ve:
            return jsonify({"error": f"Label tidak dikenali: {str(ve)}"}), 400

        model = load_model(MODEL_PATH)
        y_pred = np.argmax(model.predict(X, batch_size=32), axis=1)
        y_pred_labels = le.inverse_transform(y_pred)

        df_clean["Asli"] = y_true_labels
        df_clean["Status"] = y_pred_labels

        # Simpan hasil evaluasi ke file CSV
        df_clean.to_csv(RESULTS_PATH, index=False)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        report = classification_report(y_true, y_pred, output_dict=True)
        matrix = confusion_matrix(y_true, y_pred).tolist()
        labels = le.classes_.tolist()

        return jsonify(
            {
                "message": f"Evaluasi selesai. {skipped_rows} baris dilewati.",
                "skipped_rows": skipped_rows,
                "data": df_clean.to_dict(orient="records"),
                "classification_report": report,
                "confusion_matrix": matrix,
                "labels": labels,
                "metrics": {
                    "accuracy": round(accuracy * 100, 2),
                    "precision": round(float(precision) * 100, 2),
                    "recall": round(float(recall) * 100, 2),
                    "f1_score": round(float(f1) * 100, 2),
                },
            }
        )

    except Exception:
        import sys

        print("=== ERROR TERJADI SAAT EVALUASI ===", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return jsonify(
            {"error": "Gagal evaluasi file. Lihat log server untuk detailnya."}
        ), 500


@evaluasi_bp.route("/evaluate", methods=["POST"])
def evaluate():
    file = request.files.get("file")
    if not file or not getattr(file, "filename", "").lower().endswith(".csv"):
        return jsonify({"error": "File harus berformat .csv"}), 400

    try:
        file.save(EVAL_PATH)
        return evaluate_data(EVAL_PATH)
    except Exception:
        import sys

        print("=== ERROR TERJADI SAAT MENYIMPAN FILE ===", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return jsonify(
            {"error": "Gagal menyimpan file. Lihat log server untuk detailnya."}
        ), 500


@evaluasi_bp.route("/get_sampled_data", methods=["GET"])
def get_sampled_data():
    """Mengambil data sampel yang tersimpan"""
    if not os.path.exists(SAMPLED_DATA_PATH):
        return jsonify({"error": "Tidak ada data sampel tersimpan"}), 404

    try:
        df = pd.read_csv(SAMPLED_DATA_PATH)
        return jsonify({"data": df.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"error": f"Gagal memuat data sampel: {str(e)}"}), 500


@evaluasi_bp.route("/get_data_info", methods=["GET"])
def get_data_info():
    """Mendapatkan informasi tentang data yang tersimpan"""
    if not os.path.exists(EVAL_PATH):
        return jsonify({"error": "Tidak ada data tersimpan"}), 404

    try:
        df = pd.read_csv(EVAL_PATH)
        label_column = df.columns[-1]
        label_counts = df[label_column].value_counts().to_dict()

        return jsonify(
            {
                "total_data": len(df),
                "label_counts": label_counts,
                "labels": list(label_counts.keys()),
            }
        )
    except Exception as e:
        return jsonify({"error": f"Gagal memuat informasi data: {str(e)}"}), 500


@evaluasi_bp.route("/get_saved_data", methods=["GET"])
def get_saved_data():
    """Mengambil data evaluasi yang tersimpan"""
    if not os.path.exists(RESULTS_PATH):
        return jsonify({"error": "Tidak ada data tersimpan"}), 404

    try:
        df = pd.read_csv(RESULTS_PATH)
        return jsonify({"data": df.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"error": f"Gagal memuat data: {str(e)}"}), 500


@evaluasi_bp.route("/clear", methods=["POST"])
def clear_data():
    try:
        for path in [EVAL_PATH, RESULTS_PATH, SAMPLED_DATA_PATH]:
            if os.path.exists(path):
                os.remove(path)
        return jsonify({"message": "🧹 Data berhasil dihapus."})
    except Exception as e:
        return jsonify({"error": f"Gagal menghapus: {str(e)}"}), 500


@evaluasi_bp.route("/download/<filename>")
def download_file(filename):
    safe_files = {
        "data_eval.csv": EVAL_PATH,
        "hasil_evaluasi.csv": RESULTS_PATH,
        "data_sampled.csv": SAMPLED_DATA_PATH,
        "model_mlp_custom.keras": MODEL_PATH,
        "label_encoder.pkl": LABEL_ENCODER_PATH,
    }
    path = safe_files.get(filename)
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({"error": f"File '{filename}' tidak ditemukan."}), 404

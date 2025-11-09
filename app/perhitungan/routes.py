import os
import traceback
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend untuk server
import matplotlib.pyplot as plt
from flask import Blueprint, request, jsonify, render_template, send_file
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

perhitungan_bp = Blueprint(
    "perhitungan",
    __name__,
    url_prefix="/perhitungan",
    template_folder="templates",
    static_folder="static",
)

# Folder utama
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT_UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "perhitungan")
os.makedirs(ROOT_UPLOAD_FOLDER, exist_ok=True)

# Jalur model & encoder
MODEL_PATH = os.path.join(ROOT_UPLOAD_FOLDER, "model_mlp_custom.keras")
LABEL_ENCODER_PATH = os.path.join(ROOT_UPLOAD_FOLDER, "label_encoder.pkl")
HASIL_CSV_PATH = os.path.join(ROOT_UPLOAD_FOLDER, "hasil_perhitungan.csv")
LEARNING_CURVE_PATH = os.path.join(ROOT_UPLOAD_FOLDER, "learning_curve.png")  # Path untuk gambar learning curve

model = None
le = None


def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if "Status" not in df.columns:
        raise ValueError("Kolom 'Status' tidak ditemukan di CSV.")

    # Fitur (TF-IDF hasil preprocessing sebelumnya)
    X = df.drop(columns=["Status"]).to_numpy()

    # Label encoding
    y_raw = df["Status"].astype(str).to_numpy()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    if len(np.unique(y_encoded)) < 2:
        raise ValueError("Data harus memiliki minimal 2 kelas berbeda.")

    # Simpan encoder otomatis
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    return X, y_encoded, label_encoder, len(np.unique(y_encoded)), df


def build_and_train_model(X, y, output_dim):
    model = Sequential(
        [
            Input(shape=(X.shape[1],)),
            Dense(64, activation="relu", kernel_initializer=RandomUniform(-0.12, 0.12)),
            Dense(32, activation="relu"),
            Dense(output_dim, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=["accuracy"]
    )
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # Simpan model otomatis
    model.save(MODEL_PATH)
    return model


def analyze_neuron_effect(X, y, output_dim):
    """Metode 2: Learning Curve Analysis untuk menganalisis pengaruh jumlah neuron"""
    neuron_counts = [4, 8, 16, 32, 64, 128]  # Range neuron yang akan diuji
    train_scores = []
    val_scores = []
    best_val_acc = 0
    best_neurons = None
    
    # Hapus file learning curve sebelumnya jika ada
    if os.path.exists(LEARNING_CURVE_PATH):
        os.remove(LEARNING_CURVE_PATH)
    
    print(f"Memulai analisis learning curve dengan {len(neuron_counts)} konfigurasi...")
    
    for i, neurons in enumerate(neuron_counts):
        print(f"Testing konfigurasi {i+1}/{len(neuron_counts)}: {neurons} neuron")
        
        try:
            # Bangun model dengan jumlah neuron yang berbeda
            model = Sequential([
                Input(shape=(X.shape[1],)),
                Dense(neurons, activation='relu', kernel_regularizer=l2(0.001)),
                Dropout(0.3),
                Dense(max(neurons//2, 2), activation='relu', kernel_regularizer=l2(0.001)),  # Minimal 2 neuron
                Dropout(0.3),
                Dense(output_dim, activation='softmax'),
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=SparseCategoricalCrossentropy(),
                metrics=['accuracy']
            )
            
            # Train dengan validation split
            history = model.fit(
                X, y, 
                epochs=15, 
                batch_size=16,
                validation_split=0.2,
                verbose=0
            )
            
            # Catat skor terbaik
            train_acc = max(history.history['accuracy'])
            val_acc = max(history.history['val_accuracy'])
            
            train_scores.append(train_acc)
            val_scores.append(val_acc)
            
            # Update konfigurasi terbaik
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_neurons = neurons
                
        except Exception as e:
            print(f"Error pada konfigurasi {neurons} neuron: {str(e)}")
            # Jika error, gunakan nilai default
            train_scores.append(0.5)
            val_scores.append(0.5)
    
    # Buat visualisasi learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(neuron_counts, train_scores, 'bo-', label='Training Accuracy', linewidth=2, markersize=8)
    plt.plot(neuron_counts, val_scores, 'ro-', label='Validation Accuracy', linewidth=2, markersize=8)
    plt.xlabel('Jumlah Neuron di Layer 1')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve: Pengaruh Jumlah Neuron terhadap Performa Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Highlight titik optimal
    if best_neurons is not None:
        best_idx = neuron_counts.index(best_neurons)
        plt.scatter(best_neurons, val_scores[best_idx], color='green', s=200, zorder=5, 
                   label=f'Optimal: {best_neurons} neuron (val_acc: {val_scores[best_idx]:.3f})')
        plt.axvline(x=best_neurons, color='green', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Simpan gambar
    plt.savefig(LEARNING_CURVE_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Analisis selesai. Neuron optimal: {best_neurons}, Validation Accuracy: {best_val_acc:.4f}")
    
    return {
        "neuron_counts": neuron_counts,
        "train_scores": [float(score) for score in train_scores],
        "val_scores": [float(score) for score in val_scores],
        "best_neurons": best_neurons,
        "best_val_acc": float(best_val_acc)
    }


@perhitungan_bp.route("/")
def index():
    return render_template("perhitungan.html", page_name="perhitungan")


@perhitungan_bp.route("/check-result", methods=["GET"])
def check_result():
    """Memeriksa apakah ada hasil perhitungan yang tersimpan"""
    has_result = (
        os.path.exists(HASIL_CSV_PATH) and
        os.path.exists(MODEL_PATH) and 
        os.path.exists(LABEL_ENCODER_PATH)
    )
    
    has_learning_curve = os.path.exists(LEARNING_CURVE_PATH)
    
    return jsonify({
        "has_result": has_result,
        "has_learning_curve": has_learning_curve
    })


@perhitungan_bp.route("/analyze-neurons", methods=["POST"])
def analyze_neurons():
    """Endpoint baru untuk analisis learning curve"""
    global model, le
    
    file = request.files.get("file")
    if not file or not file.filename or not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "File harus CSV."}), 400

    try:
        csv_path = os.path.join(ROOT_UPLOAD_FOLDER, file.filename)
        file.save(csv_path)

        # Load data
        X, y, encoder, output_dim, df_clean = load_and_prepare_data(csv_path)
        le = encoder

        # Lakukan analisis learning curve
        analysis_result = analyze_neuron_effect(X, y, output_dim)
        
        # Konversi numpy types ke Python native types untuk JSON
        analysis_result["neuron_counts"] = [int(x) for x in analysis_result["neuron_counts"]]
        analysis_result["train_scores"] = [float(x) for x in analysis_result["train_scores"]]
        analysis_result["val_scores"] = [float(x) for x in analysis_result["val_scores"]]
        analysis_result["best_neurons"] = int(analysis_result["best_neurons"])
        analysis_result["best_val_acc"] = float(analysis_result["best_val_acc"])

        return jsonify({
            "message": "Analisis learning curve selesai.",
            "analysis": analysis_result,
            "learning_curve_url": f"/perhitungan/download/learning_curve.png"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal menganalisis: {str(e)}"}), 500


@perhitungan_bp.route("/process", methods=["POST"])
def process_csv():
    global model, le

    # Pertama, cek apakah sudah ada hasil yang tersimpan
    if (
        os.path.exists(HASIL_CSV_PATH)
        and os.path.exists(MODEL_PATH)
        and os.path.exists(LABEL_ENCODER_PATH)
    ):
        try:
            # Langsung muat hasil yang tersimpan (SEMUA DATA)
            df = pd.read_csv(HASIL_CSV_PATH, encoding="utf-8-sig")

            # Muat model dan encoder
            model = load_model(MODEL_PATH)
            with open(LABEL_ENCODER_PATH, "rb") as f:
                le = pickle.load(f)

            # Hitung metrik evaluasi menggunakan SEMUA DATA
            if "Status" in df.columns and "Prediksi" in df.columns:
                y_actual = df["Status"].astype(str)
                y_pred = df["Prediksi"].astype(str)

                acc = accuracy_score(y_actual, y_pred)
                prec = precision_score(
                    y_actual, y_pred, average="weighted", zero_division=0
                )
                rec = recall_score(
                    y_actual, y_pred, average="weighted", zero_division=0
                )
                f1 = f1_score(y_actual, y_pred, average="weighted", zero_division=0)
                cm = confusion_matrix(
                    y_actual, y_pred, labels=np.unique(y_actual)
                ).tolist()
                labels = list(np.unique(y_actual))

                # HANYA TAMPILKAN 1000 BARIS PERTAMA DI FRONTEND
                df_display = df.head(1000)

                return jsonify(
                    {
                        "message": "Data hasil sebelumnya berhasil dimuat.",
                        "train": df_display.to_dict(orient="records"),
                        "accuracy": round(float(acc) * 100, 2),
                        "precision": round(float(prec) * 100, 2),
                        "recall": round(float(rec) * 100, 2),
                        "f1": round(float(f1) * 100, 2),
                        "labels": labels,
                        "confusion": cm,
                        "total_records": len(df),
                        "displayed_records": len(df_display),
                    }
                )
            else:
                df_display = df.head(1000)
                return jsonify(
                    {
                        "message": "Data hasil sebelumnya berhasil dimuat.",
                        "train": df_display.to_dict(orient="records"),
                        "total_records": len(df),
                        "displayed_records": len(df_display),
                    }
                )

        except Exception:
            traceback.print_exc()
            # Jangan return error, biarkan lanjut ke proses normal

    # Jika tidak ada hasil tersimpan, lanjutkan dengan proses normal
    file = request.files.get("file")
    if not file or not file.filename or not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "File harus CSV."}), 400

    try:
        csv_path = os.path.join(ROOT_UPLOAD_FOLDER, file.filename)
        file.save(csv_path)

        X, y, encoder, output_dim, df_clean = load_and_prepare_data(csv_path)

        model = build_and_train_model(X, y, output_dim)
        le = encoder

        # Prediksi training set (hanya evaluasi internal) menggunakan SEMUA DATA
        y_pred = le.inverse_transform(np.argmax(model.predict(X), axis=1))
        y_actual = le.inverse_transform(y)

        df_clean["Prediksi"] = y_pred

        # Simpan hasil ke CSV (SEMUA DATA)
        df_clean.to_csv(HASIL_CSV_PATH, index=False, encoding="utf-8-sig")

        # Hitung metrik evaluasi menggunakan SEMUA DATA
        acc = accuracy_score(y_actual, y_pred)
        prec = precision_score(y_actual, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_actual, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_actual, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_actual, y_pred, labels=np.unique(y_actual)).tolist()
        labels = list(np.unique(y_actual))

        # HANYA TAMPILKAN 1000 BARIS PERTAMA DI FRONTEND
        df_display = df_clean.head(1000)

        return jsonify(
            {
                "message": "Pelatihan selesai.",
                "train": df_display.to_dict(orient="records"),
                "accuracy": round(float(acc) * 100, 2),
                "precision": round(float(prec) * 100, 2),
                "recall": round(float(rec) * 100, 2),
                "f1": round(float(f1) * 100, 2),
                "labels": labels,
                "confusion": cm,
                "total_records": len(df_clean),
                "displayed_records": len(df_display),
            }
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses: {str(e)}"}), 500


@perhitungan_bp.route("/load-result", methods=["GET"])
def load_result():
    """Memuat hasil perhitungan yang tersimpan"""
    if not os.path.exists(HASIL_CSV_PATH):
        return jsonify({"error": "Tidak ada data hasil yang tersimpan."}), 404

    try:
        # Muat SEMUA DATA dari CSV
        df = pd.read_csv(HASIL_CSV_PATH, encoding="utf-8-sig")

        # Hitung metrik evaluasi menggunakan SEMUA DATA
        if "Status" in df.columns and "Prediksi" in df.columns:
            y_actual = df["Status"].astype(str)
            y_pred = df["Prediksi"].astype(str)

            acc = accuracy_score(y_actual, y_pred)
            prec = precision_score(
                y_actual, y_pred, average="weighted", zero_division=0
            )
            rec = recall_score(y_actual, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_actual, y_pred, average="weighted", zero_division=0)
            cm = confusion_matrix(y_actual, y_pred, labels=np.unique(y_actual)).tolist()
            labels = list(np.unique(y_actual))

            # HANYA TAMPILKAN 1000 BARIS PERTAMA DI FRONTEND
            df_display = df.head(1000)

            return jsonify(
                {
                    "train": df_display.to_dict(orient="records"),
                    "accuracy": round(float(acc) * 100, 2),
                    "precision": round(float(prec) * 100, 2),
                    "recall": round(float(rec) * 100, 2),
                    "f1": round(float(f1) * 100, 2),
                    "labels": labels,
                    "confusion": cm,
                    "total_records": len(df),
                    "displayed_records": len(df_display),
                }
            )
        else:
            df_display = df.head(1000)
            return jsonify(
                {
                    "train": df_display.to_dict(orient="records"),
                    "message": "Data hasil dimuat, tetapi tidak memiliki kolom yang diperlukan untuk evaluasi.",
                    "total_records": len(df),
                    "displayed_records": len(df_display),
                }
            )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memuat data hasil: {str(e)}"}), 500


@perhitungan_bp.route("/download/<filename>")
def download_file(filename):
    safe_files = {
        "model_mlp_custom.keras": MODEL_PATH,
        "label_encoder.pkl": LABEL_ENCODER_PATH,
        "hasil_perhitungan.csv": HASIL_CSV_PATH,
        "learning_curve.png": LEARNING_CURVE_PATH,  # Tambahkan learning curve
    }
    path = safe_files.get(filename)
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({"error": f"File '{filename}' tidak ditemukan."}), 404


@perhitungan_bp.route("/clear", methods=["POST"])
def clear_data():
    global model, le
    files_to_clear = [MODEL_PATH, LABEL_ENCODER_PATH, HASIL_CSV_PATH, LEARNING_CURVE_PATH]
    for path in files_to_clear:
        if os.path.exists(path):
            os.remove(path)
    model, le = None, None
    return jsonify({"message": "Semua data dan model berhasil dihapus."})


@perhitungan_bp.route("/copy-download-model")
def copy_and_download_model():
    try:
        return send_file(MODEL_PATH, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"Gagal mendownload model: {str(e)}"}), 500


@perhitungan_bp.route("/copy-download-encoder")
def copy_and_download_encoder():
    try:
        return send_file(LABEL_ENCODER_PATH, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"Gagal mendownload encoder: {str(e)}"}), 500
import os
import traceback
import numpy as np
import pandas as pd
import pickle
import time
import psutil
import gc
import random
from datetime import datetime
from flask import Blueprint, request, jsonify, render_template, send_file
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, Callback
import tensorflow as tf

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

# JALUR CACHE BARU UNTUK VISUALISASI
ARCHITECTURE_RESULTS_CACHE = os.path.join(
    ROOT_UPLOAD_FOLDER, "architecture_results_cache.pkl"
)
TRAINING_HISTORIES_CACHE = os.path.join(
    ROOT_UPLOAD_FOLDER, "training_histories_cache.pkl"
)
COMPARISON_DATA_CACHE = os.path.join(ROOT_UPLOAD_FOLDER, "comparison_data_cache.pkl")
FULL_RESULTS_CACHE = os.path.join(ROOT_UPLOAD_FOLDER, "full_results_cache.pkl")

model = None
le = None


# Fungsi bantu untuk rounding yang aman
def safe_round(value, decimals=2):
    """Round value safely, handling None and NaN"""
    if value is None:
        return 0.0
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return 0.0


# Fungsi bantu untuk confusion matrix yang aman
def safe_confusion_matrix(cm):
    """Convert confusion matrix to list safely"""
    if cm is None:
        return []
    if hasattr(cm, "tolist"):
        return cm.tolist()
    if isinstance(cm, (list, np.ndarray)):
        return cm
    return []


# Set fixed seeds untuk reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class MemoryCallback(Callback):
    """Callback untuk melacak penggunaan memori"""

    def __init__(self):
        super().__init__()
        self.memory_usage = []

    def on_epoch_begin(self, epoch, logs=None):
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(memory_mb)


# =============================================
# SISTEM CACHE UNTUK VISUALISASI - DIPERBAIKI
# =============================================


def get_cache_status():
    """Mendapatkan status lengkap semua file cache"""
    cache_files = {
        "architecture_results": ARCHITECTURE_RESULTS_CACHE,
        "training_histories": TRAINING_HISTORIES_CACHE,
        "comparison_data": COMPARISON_DATA_CACHE,
        "full_results": FULL_RESULTS_CACHE,
        "model": MODEL_PATH,
        "encoder": LABEL_ENCODER_PATH,
        "hasil_csv": HASIL_CSV_PATH,
    }

    status = {}
    for name, path in cache_files.items():
        exists = os.path.exists(path)
        status[name] = {
            "exists": exists,
            "path": os.path.basename(path),
            "size": os.path.getsize(path) if exists else 0,
            "modified": datetime.fromtimestamp(os.path.getmtime(path)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if exists
            else None,
        }

    return status


def debug_cache_contents():
    """Debug function untuk melihat isi cache"""
    cache_status = get_cache_status()
    print("\n=== DEBUG CACHE CONTENTS ===")
    for name, info in cache_status.items():
        status = "✅ ADA" if info["exists"] else "❌ TIDAK ADA"
        print(f"{name}: {status}")
        if info["exists"]:
            print(f"  Size: {info['size']} bytes")
            print(f"  Modified: {info['modified']}")

    # Coba baca isi cache files
    try:
        if os.path.exists(ARCHITECTURE_RESULTS_CACHE):
            with open(ARCHITECTURE_RESULTS_CACHE, "rb") as f:
                arch_data = pickle.load(f)
                print(
                    f"Architecture results: {len(arch_data) if arch_data else 0} items"
                )

        if os.path.exists(TRAINING_HISTORIES_CACHE):
            with open(TRAINING_HISTORIES_CACHE, "rb") as f:
                hist_data = pickle.load(f)
                print(
                    f"Training histories: {len(hist_data) if hist_data else 0} models"
                )

        if os.path.exists(COMPARISON_DATA_CACHE):
            with open(COMPARISON_DATA_CACHE, "rb") as f:
                comp_data = pickle.load(f)
                print(
                    f"Comparison data keys: {list(comp_data.keys()) if comp_data else 'None'}"
                )

        if os.path.exists(FULL_RESULTS_CACHE):
            with open(FULL_RESULTS_CACHE, "rb") as f:
                full_data = pickle.load(f)
                print(f"Full results type: {type(full_data)}")
                if isinstance(full_data, dict):
                    print(f"Full results keys: {list(full_data.keys())}")
        else:
            print("Full results cache: TIDAK DITEMUKAN")

    except Exception as e:
        print(f"Error reading cache: {e}")

    print("=== END DEBUG ===\n")


def save_comprehensive_cache(
    architecture_results, training_histories, comparison_data, full_results=None
):
    """Menyimpan semua data untuk visualisasi ke cache - VERSI DIPERBAIKI"""
    try:
        print("💾 Menyimpan data ke cache...")

        # Simpan architecture results
        with open(ARCHITECTURE_RESULTS_CACHE, "wb") as f:
            pickle.dump(architecture_results, f)
        print(f"  ✅ architecture_results: {len(architecture_results)} items")

        # Simpan training histories (data untuk chart)
        with open(TRAINING_HISTORIES_CACHE, "wb") as f:
            pickle.dump(training_histories, f)
        print(f"  ✅ training_histories: {len(training_histories)} models")

        # Simpan comparison data (untuk tabel perbandingan)
        with open(COMPARISON_DATA_CACHE, "wb") as f:
            pickle.dump(comparison_data, f)
        print(f"  ✅ comparison_data: {len(comparison_data)} keys")

        # Simpan full results - PASTIKAN INI DISIMPAN
        full_data = {
            "architecture_results": architecture_results,
            "training_histories": training_histories,
            "comparison_data": comparison_data,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
            "total_architectures": len(architecture_results),
            "cache_info": "Full results cache for visualization",
        }

        with open(FULL_RESULTS_CACHE, "wb") as f:
            pickle.dump(full_data, f)
        print(f"  ✅ full_results: Disimpan dengan {len(full_data)} keys")

        # Debug untuk memastikan
        debug_cache_contents()

        print("✅ Semua data visualisasi berhasil disimpan ke cache")
        return True
    except Exception as e:
        print(f"❌ Gagal menyimpan cache visualisasi: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def load_comprehensive_cache():
    """Memuat semua data visualisasi dari cache - VERSI DIPERBAIKI"""
    try:
        architecture_results = None
        training_histories = None
        comparison_data = None
        full_results = None

        print("📂 Memuat data dari cache...")

        if os.path.exists(ARCHITECTURE_RESULTS_CACHE):
            with open(ARCHITECTURE_RESULTS_CACHE, "rb") as f:
                architecture_results = pickle.load(f)
            print(
                f"  ✅ architecture_results: {len(architecture_results) if architecture_results else 0} items"
            )

        if os.path.exists(TRAINING_HISTORIES_CACHE):
            with open(TRAINING_HISTORIES_CACHE, "rb") as f:
                training_histories = pickle.load(f)
            print(
                f"  ✅ training_histories: {len(training_histories) if training_histories else 0} models"
            )

        if os.path.exists(COMPARISON_DATA_CACHE):
            with open(COMPARISON_DATA_CACHE, "rb") as f:
                comparison_data = pickle.load(f)
            print(
                f"  ✅ comparison_data: {len(comparison_data) if comparison_data else 0} keys"
            )

        if os.path.exists(FULL_RESULTS_CACHE):
            with open(FULL_RESULTS_CACHE, "rb") as f:
                full_results = pickle.load(f)
            print(
                f"  ✅ full_results: {len(full_results) if isinstance(full_results, dict) else 'invalid'} keys"
            )
        else:
            print("  ❌ full_results: File tidak ditemukan")

        # Jika full_results tidak ada, buat dari data individual
        if (
            not full_results
            and architecture_results
            and training_histories
            and comparison_data
        ):
            full_results = {
                "architecture_results": architecture_results,
                "training_histories": training_histories,
                "comparison_data": comparison_data,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0-reconstructed",
                "total_architectures": len(architecture_results),
                "cache_info": "Reconstructed from individual cache files",
            }
            print("  🔄 full_results: Direkonstruksi dari data individual")

        print("✅ Data visualisasi berhasil dimuat dari cache")
        return architecture_results, training_histories, comparison_data, full_results
    except Exception as e:
        print(f"❌ Gagal memuat cache visualisasi: {str(e)}")
        import traceback

        traceback.print_exc()
        return None, None, None, None


def prepare_architecture_comparison_data(architecture_results):
    """Mempersiapkan data untuk chart dan tabel perbandingan - DIPERBAIKI URUTAN"""
    if not architecture_results:
        return None

    # Urutkan berdasarkan urutan yang diinginkan: A, B, C, D
    desired_order = [
        "MLP-A (Sederhana)",
        "MLP-B (Sedang)",
        "MLP-C (Dalam)",
        "MLP-D (Wide + Shallow)",
    ]

    # Buat mapping untuk akses cepat
    result_mapping = {result["architecture"]: result for result in architecture_results}

    # Siapkan data dalam urutan yang benar
    ordered_results = []
    for arch_name in desired_order:
        if arch_name in result_mapping:
            ordered_results.append(result_mapping[arch_name])
        else:
            # Tambahkan placeholder jika arsitektur tidak ada
            ordered_results.append(
                {
                    "architecture": arch_name,
                    "val_accuracy": 0,
                    "macro_f1": 0,
                    "macro_precision": 0,
                    "macro_recall": 0,
                    "training_time_total": 0,
                    "avg_memory_mb": 0,
                    "inference_time_ms": 0,
                    "total_params": 0,
                    "epochs_used": 0,
                    "success": False,
                }
            )

    comparison_data = {
        "labels": [result["architecture"] for result in ordered_results],
        "accuracy": [result["val_accuracy"] for result in ordered_results],
        "macro_f1": [result["macro_f1"] for result in ordered_results],
        "macro_precision": [
            result.get("macro_precision", 0) for result in ordered_results
        ],
        "macro_recall": [result.get("macro_recall", 0) for result in ordered_results],
        "training_time": [result["training_time_total"] for result in ordered_results],
        "memory_usage": [result["avg_memory_mb"] for result in ordered_results],
        "inference_time": [
            result.get("inference_time_ms", 0) for result in ordered_results
        ],
        "parameters": [result["total_params"] for result in ordered_results],
        "epochs_used": [result["epochs_used"] for result in ordered_results],
    }

    return comparison_data


def validate_cache_completeness():
    """Validasi kelengkapan cache - VERSI DIPERBAIKI"""
    required_files = [
        ARCHITECTURE_RESULTS_CACHE,
        TRAINING_HISTORIES_CACHE,
        COMPARISON_DATA_CACHE,
        MODEL_PATH,
        LABEL_ENCODER_PATH,
        HASIL_CSV_PATH,
    ]

    # FULL_RESULTS_CACHE tidak wajib karena bisa direkonstruksi
    optional_files = [FULL_RESULTS_CACHE]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(os.path.basename(file_path))

    # Cek juga optional files untuk informasi
    for file_path in optional_files:
        if not os.path.exists(file_path):
            print(f"⚠️ Optional file missing: {os.path.basename(file_path)}")

    return len(missing_files) == 0, missing_files


def clear_visualization_cache():
    """Menghapus cache visualisasi"""
    try:
        cache_files = [
            ARCHITECTURE_RESULTS_CACHE,
            TRAINING_HISTORIES_CACHE,
            COMPARISON_DATA_CACHE,
            FULL_RESULTS_CACHE,
        ]

        deleted_files = []
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                deleted_files.append(os.path.basename(cache_file))
                print(f"✅ Menghapus: {os.path.basename(cache_file)}")

        print("✅ Cache visualisasi berhasil dihapus")
        return True, deleted_files
    except Exception as e:
        print(f"❌ Gagal menghapus cache visualisasi: {str(e)}")
        return False, []


# =============================================
# FUNGSI MLP ARCHITECTURES
# =============================================


def load_and_prepare_data(csv_path):
    """Memuat dan mempersiapkan data dari CSV"""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

        if "Status" not in df.columns:
            raise ValueError("Kolom 'Status' tidak ditemukan di CSV.")

        # Fitur (TF-IDF hasil preprocessing sebelumnya) - tidak perlu scaling
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

    except Exception as e:
        raise ValueError(f"Gagal memuat data: {str(e)}")


def count_model_parameters(model):
    """Menghitung total parameter dalam model"""
    trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable_params = np.sum(
        [np.prod(v.get_shape()) for v in model.non_trainable_weights]
    )
    return int(trainable_params + non_trainable_params)


def create_mlp_a(input_dim, output_dim, dropout_rate=0.3, l2_rate=1e-5):
    """MLP-A (Sederhana): 128 neuron, ReLU, Dropout"""
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(
                128,
                activation="relu",
                kernel_initializer=GlorotUniform(),
                kernel_regularizer=l2(l2_rate),
            ),
            Dropout(dropout_rate),
            Dense(output_dim, activation="softmax"),
        ]
    )
    return model, "MLP-A (Sederhana)"


def create_mlp_b(input_dim, output_dim, dropout_rate=0.4, l2_rate=1e-5):
    """MLP-B (Sedang): 256 → 128 neuron, BatchNorm, Dropout"""
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(256, activation="relu", kernel_regularizer=l2(l2_rate)),
            BatchNormalization(),
            Dense(128, activation="relu", kernel_regularizer=l2(l2_rate)),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(output_dim, activation="softmax"),
        ]
    )
    return model, "MLP-B (Sedang)"


def create_mlp_c(input_dim, output_dim, dropout_rate=0.5, l2_rate=1e-5):
    """MLP-C (Dalam): 512 → 256 → 128 neuron, L2 regularization, Dropout"""
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(512, activation="relu", kernel_regularizer=l2(l2_rate)),
            Dense(256, activation="relu", kernel_regularizer=l2(l2_rate)),
            Dense(128, activation="relu", kernel_regularizer=l2(l2_rate)),
            Dropout(dropout_rate),
            Dense(output_dim, activation="softmax"),
        ]
    )
    return model, "MLP-C (Dalam)"


def create_mlp_d(input_dim, output_dim, dropout_rate=0.4, l2_rate=1e-5):
    """MLP-D (Wide + shallow): 1024 neuron, Dropout"""
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(1024, activation="relu", kernel_regularizer=l2(l2_rate)),
            Dropout(dropout_rate),
            Dense(output_dim, activation="softmax"),
        ]
    )
    return model, "MLP-D (Wide + Shallow)"


def get_optimizer(optimizer_name, learning_rate):
    """Mengembalikan optimizer berdasarkan nama dan learning rate"""
    if optimizer_name == "adam":
        return Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
    elif optimizer_name == "sgd":
        return SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        return Adam(learning_rate=learning_rate)


def evaluate_architecture_with_hyperparams(
    X, y, output_dim, architecture_func, architecture_name, hyperparams
):
    """Mengevaluasi arsitektur dengan hyperparameter spesifik"""
    try:
        print(f"🧪 Mengevaluasi {architecture_name} dengan hyperparams: {hyperparams}")

        # Set seeds untuk reproducibility
        set_seeds(42)

        # Buat model dengan hyperparameter
        model, name = architecture_func(
            X.shape[1],
            output_dim,
            dropout_rate=hyperparams["dropout_rate"],
            l2_rate=hyperparams["l2_rate"],
        )

        # Dapatkan optimizer
        optimizer = get_optimizer(
            hyperparams["optimizer"], hyperparams["learning_rate"]
        )

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        # Callbacks
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=0
        )
        memory_callback = MemoryCallback()

        # Hitung parameter
        total_params = count_model_parameters(model)

        # Training dengan pengukuran waktu
        start_time = time.time()
        history = model.fit(
            X,
            y,
            epochs=50,
            batch_size=hyperparams["batch_size"],
            validation_split=0.2,
            callbacks=[early_stop, memory_callback],
            verbose=0,
        )
        training_time = time.time() - start_time

        # Hitung epoch yang benar-benar digunakan
        actual_epochs = len(history.history["accuracy"])
        time_per_epoch = training_time / actual_epochs if actual_epochs > 0 else 0

        # Hitung metrik dengan error handling
        train_acc = max(history.history["accuracy"]) * 100
        val_acc = max(history.history["val_accuracy"]) * 100
        final_train_acc = history.history["accuracy"][-1] * 100
        final_val_acc = history.history["val_accuracy"][-1] * 100

        # Memory usage
        avg_memory = (
            np.mean(memory_callback.memory_usage) if memory_callback.memory_usage else 0
        )
        max_memory = (
            max(memory_callback.memory_usage) if memory_callback.memory_usage else 0
        )

        # Prediksi untuk metrik tambahan dengan error handling
        y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
        y_true = y

        # ========== EVALUASI MATRIX YANG DIPERBAIKI ==========
        # Inisialisasi variabel sebelum try-catch
        macro_f1 = 0.0
        macro_precision = 0.0
        macro_recall = 0.0
        accuracy_val = 0.0
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        cm = []
        inference_time_per_sample = 0.0
        class_report = {}

        try:
            # Macro metrics (utama)
            macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
            macro_precision = (
                precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
            )
            macro_recall = (
                recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
            )

            # Precision, Recall per kelas - pastikan hasilnya iterable
            precision_per_class = (
                precision_score(y_true, y_pred, average=None, zero_division=0) * 100
            )
            recall_per_class = (
                recall_score(y_true, y_pred, average=None, zero_division=0) * 100
            )
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0) * 100

            # Accuracy (sekunder)
            accuracy_val = accuracy_score(y_true, y_pred) * 100

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Inference time measurement
            inference_start = time.time()
            # Predict multiple times for more accurate measurement
            for _ in range(10):
                model.predict(X[:100], verbose=0)  # Use subset for speed
            inference_time = (time.time() - inference_start) / 10  # Average time
            inference_time_per_sample = (
                inference_time / 100
            ) * 1000  # Convert to ms per sample

            # Classification report untuk detail
            class_report = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )

        except Exception as e:
            print(f"⚠️  Error calculating metrics for {architecture_name}: {e}")

        # Hasil tanpa menyertakan objek model
        result = {
            "architecture": name,
            "hyperparams": hyperparams,
            "total_params": total_params,
            "training_time_total": safe_round(training_time),
            "time_per_epoch": safe_round(time_per_epoch),
            "epochs_used": actual_epochs,
            "train_accuracy": safe_round(train_acc),
            "val_accuracy": safe_round(val_acc),
            "final_train_accuracy": safe_round(final_train_acc),
            "final_val_accuracy": safe_round(final_val_acc),
            "avg_memory_mb": safe_round(avg_memory),
            "max_memory_mb": safe_round(max_memory),
            # Metrik evaluasi baru dengan handling yang aman
            "macro_f1": safe_round(macro_f1),
            "macro_precision": safe_round(macro_precision),  # BARU
            "macro_recall": safe_round(macro_recall),  # BARU
            "accuracy": safe_round(accuracy_val),
            "precision_per_class": [safe_round(p) for p in precision_per_class]
            if isinstance(precision_per_class, (list, np.ndarray))
            else [],
            "recall_per_class": [safe_round(r) for r in recall_per_class]
            if isinstance(recall_per_class, (list, np.ndarray))
            else [],
            "f1_per_class": [safe_round(f) for f in f1_per_class]
            if isinstance(f1_per_class, (list, np.ndarray))
            else [],
            "inference_time_ms": safe_round(inference_time_per_sample, 4),
            "confusion_matrix": safe_confusion_matrix(cm),
            "classification_report": class_report,
            "success": True,
        }

        print(
            f"   ✅ {architecture_name}: Val Acc={result['val_accuracy']}%, Macro F1={result['macro_f1']}%, Inference={result['inference_time_ms']}ms"
        )

        # Clean up memory
        del history
        gc.collect()

        # Kembalikan result dan model secara terpisah
        return result, model

    except Exception as e:
        print(
            f"   ❌ {architecture_name} dengan hyperparams {hyperparams} gagal: {str(e)}"
        )
        error_result = {
            "architecture": architecture_name,
            "hyperparams": hyperparams,
            "success": False,
            "error": str(e),
        }
        return error_result, None


def generate_hyperparameter_combinations():
    """Generate semua kombinasi hyperparameter untuk grid search"""
    optimizers = ["adam", "sgd"]
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
    batch_sizes = [32, 64, 128]
    dropout_rates = [0.2, 0.3, 0.4, 0.5]
    l2_rates = [1e-6, 1e-5, 1e-4, 1e-3]

    combinations = []
    for optimizer in optimizers:
        for lr in learning_rates:
            for bs in batch_sizes:
                for dropout in dropout_rates:
                    for l2_rate in l2_rates:
                        combinations.append(
                            {
                                "optimizer": optimizer,
                                "learning_rate": lr,
                                "batch_size": bs,
                                "dropout_rate": dropout,
                                "l2_rate": l2_rate,
                            }
                        )

    return combinations


def perform_hyperparameter_tuning(
    X, y, output_dim, architecture_func, architecture_name, max_combinations=20
):
    """Melakukan hyperparameter tuning dengan grid search terbatas"""
    print(f"🎯 Memulai Hyperparameter Tuning untuk {architecture_name}")

    # Generate semua kombinasi
    all_combinations = generate_hyperparameter_combinations()

    # Pilih kombinasi secara acak tapi reproducible
    random.seed(42)
    selected_combinations = random.sample(
        all_combinations, min(max_combinations, len(all_combinations))
    )

    successful_models = []  # List of tuples (result, model)

    for i, hyperparams in enumerate(selected_combinations):
        print(f"   🔍 Kombinasi {i + 1}/{len(selected_combinations)}")

        result, model = evaluate_architecture_with_hyperparams(
            X, y, output_dim, architecture_func, architecture_name, hyperparams
        )

        if result.get("success", False) and model is not None:
            successful_models.append((result, model))

        # Beri jeda untuk stabilisasi memori
        time.sleep(1)

    if not successful_models:
        raise ValueError(
            f"Tidak ada kombinasi hyperparameter yang berhasil untuk {architecture_name}"
        )

    # Urutkan berdasarkan validation accuracy
    successful_models.sort(key=lambda x: x[0]["val_accuracy"], reverse=True)

    best_result, best_model = successful_models[0]
    best_hyperparams = best_result["hyperparams"]

    # Extract hanya results untuk return
    arch_results = [result for result, _ in successful_models]

    print(f"✅ Hyperparameter terbaik untuk {architecture_name}:")
    print(f"   Val Accuracy: {best_result['val_accuracy']}%")
    print(f"   Macro F1: {best_result['macro_f1']}%")
    print(f"   Hyperparams: {best_hyperparams}")

    return best_model, best_hyperparams, arch_results


def analyze_architectures_with_tuning(X, y, output_dim):
    """Menganalisis semua arsitektur MLP dengan hyperparameter tuning - DIPERBAIKI URUTAN"""
    # Definisikan urutan arsitektur yang FIXED
    architectures = [
        (create_mlp_a, "MLP-A (Sederhana)"),
        (create_mlp_b, "MLP-B (Sedang)"),
        (create_mlp_c, "MLP-C (Dalam)"),
        (create_mlp_d, "MLP-D (Wide + Shallow)"),
    ]

    successful_architectures = []  # List of tuples (result, model, architecture_name)
    all_training_histories = {}  # Simpan training history untuk setiap model
    architecture_results = []  # Hasil untuk setiap arsitektur dalam urutan yang benar

    print("🔍 Memulai analisis arsitektur MLP dengan Hyperparameter Tuning...")

    for arch_func, arch_name in architectures:
        try:
            best_model, best_hyperparams, arch_results = perform_hyperparameter_tuning(
                X, y, output_dim, arch_func, arch_name, max_combinations=15
            )

            # Cari result yang sesuai dengan best_hyperparams
            best_arch_result = None
            for result in arch_results:
                if (
                    result["hyperparams"] == best_hyperparams
                    and result["architecture"] == arch_name
                ):
                    best_arch_result = result
                    break

            if best_arch_result and best_model is not None:
                successful_architectures.append(
                    (best_arch_result, best_model, arch_name)
                )
                architecture_results.append(best_arch_result)  # Tambahkan dalam urutan

                # Simpan training history untuk chart
                all_training_histories[arch_name] = {
                    "accuracy": [
                        best_arch_result.get("train_accuracy", 0) / 100
                    ],  # Convert to 0-1
                    "val_accuracy": [
                        best_arch_result.get("val_accuracy", 0) / 100
                    ],  # Convert to 0-1
                    "loss": [0.5],  # Default values
                    "val_loss": [0.5],
                }

                print(f"✅ {arch_name} berhasil ditambahkan ke kandidat")
            else:
                print(f"⚠️  {arch_name} memiliki masalah dengan model atau result")

            # Beri jeda antara arsitektur untuk stabilisasi memori
            time.sleep(2)

        except Exception as e:
            print(f"❌ Arsitektur {arch_name} gagal dalam tuning: {str(e)}")
            # Tambahkan result kosong untuk menjaga urutan
            empty_result = {
                "architecture": arch_name,
                "hyperparams": {},
                "total_params": 0,
                "training_time_total": 0,
                "time_per_epoch": 0,
                "epochs_used": 0,
                "train_accuracy": 0,
                "val_accuracy": 0,
                "final_train_accuracy": 0,
                "final_val_accuracy": 0,
                "avg_memory_mb": 0,
                "max_memory_mb": 0,
                "macro_f1": 0,
                "macro_precision": 0,
                "macro_recall": 0,
                "accuracy": 0,
                "precision_per_class": [],
                "recall_per_class": [],
                "f1_per_class": [],
                "inference_time_ms": 0,
                "confusion_matrix": [],
                "classification_report": {},
                "success": False,
                "error": str(e),
            }
            architecture_results.append(empty_result)
            continue

    if not successful_architectures:
        raise ValueError("Tidak ada arsitektur yang berhasil dilatih")

    # =============================================
    # 🎯 PERBAIKAN: KRITERIA PEMILIHAN TERBAIK
    # Macro F1 sebagai UTAMA, Accuracy sebagai SEKUNDER
    # =============================================

    best_result, best_model, best_architecture = None, None, None
    best_macro_f1 = -1
    best_val_accuracy = -1

    print("\n📊 Menentukan arsitektur terbaik berdasarkan:")
    print("   🎯 Primary: Macro F1 Score")
    print("   🎯 Secondary: Validation Accuracy")
    print("   📈 Tertiary: Inference Time (jika semua sama)")

    for result, model, arch_name in successful_architectures:
        current_macro_f1 = result.get("macro_f1", 0)
        current_val_accuracy = result.get("val_accuracy", 0)
        current_inference_time = result.get("inference_time_ms", float("inf"))

        print(
            f"   🔍 {arch_name}: F1={current_macro_f1}%, Acc={current_val_accuracy}%, Inference={current_inference_time}ms"
        )

        # Kriteria 1: Bandingkan Macro F1
        if current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            best_val_accuracy = current_val_accuracy
            best_result, best_model, best_architecture = result, model, arch_name
            print(f"     🆕 LEADER: F1 lebih tinggi")

        # Kriteria 2: Jika Macro F1 sama, bandingkan Validation Accuracy
        elif current_macro_f1 == best_macro_f1:
            if current_val_accuracy > best_val_accuracy:
                best_val_accuracy = current_val_accuracy
                best_result, best_model, best_architecture = result, model, arch_name
                print(f"     🆕 LEADER: Accuracy lebih tinggi (F1 sama)")

            # Kriteria 3: Jika F1 dan Accuracy sama, bandingkan Inference Time
            elif (
                current_val_accuracy == best_val_accuracy
                and current_inference_time
                < best_result.get("inference_time_ms", float("inf"))
            ):
                best_result, best_model, best_architecture = result, model, arch_name
                print(f"     🆕 LEADER: Inference lebih cepat (F1 & Acc sama)")

    if best_result:
        best_hyperparams = best_result["hyperparams"]

        print(f"\n🎯 ARSITEKTUR TERBAIK DIPILIH: {best_architecture}")
        print(f"   📊 Macro F1: {best_result['macro_f1']}% (Primary)")
        print(f"   ✅ Validation Accuracy: {best_result['val_accuracy']}% (Secondary)")
        print(f"   ⚡ Inference Time: {best_result['inference_time_ms']}ms")
        print(f"   ⚙️  Hyperparameters: {best_hyperparams}")
        print(f"   🔢 Total Parameters: {best_result['total_params']:,}")
        print(f"   ⏱️  Training Time: {best_result['training_time_total']}s")
    else:
        # Fallback jika tidak ada yang berhasil
        best_model, best_architecture, best_hyperparams = None, "Tidak ada", {}
        print("❌ Tidak ada arsitektur yang memenuhi kriteria")

    return (
        best_model,
        best_architecture,
        best_hyperparams,
        architecture_results,  # Kembalikan dalam urutan definisi
        all_training_histories,
    )


# =============================================
# ROUTES DENGAN SISTEM CACHE LENGKAP - DIPERBAIKI
# =============================================


@perhitungan_bp.route("/")
def index():
    return render_template("perhitungan.html", page_name="perhitungan")


@perhitungan_bp.route("/check-result", methods=["GET"])
def check_result():
    """Memeriksa apakah ada hasil perhitungan yang tersimpan"""
    try:
        has_model = os.path.exists(MODEL_PATH)
        has_encoder = os.path.exists(LABEL_ENCODER_PATH)
        has_result = os.path.exists(HASIL_CSV_PATH)

        # Cek juga apakah ada cache visualisasi
        cache_complete, missing_files = validate_cache_completeness()

        return jsonify(
            {
                "has_result": has_model and has_encoder and has_result,
                "has_complete_cache": cache_complete,
                "details": {
                    "model_exists": has_model,
                    "encoder_exists": has_encoder,
                    "result_exists": has_result,
                    "cache_complete": cache_complete,
                    "missing_files": missing_files if not cache_complete else [],
                },
            }
        )
    except Exception as e:
        return jsonify({"has_result": False, "error": str(e)})


@perhitungan_bp.route("/process", methods=["POST"])
def process_csv():
    global model, le

    # Cek apakah sudah ada hasil yang tersimpan DAN cache lengkap
    cache_complete, missing_files = validate_cache_completeness()

    if cache_complete:
        try:
            print("📁 Memuat hasil sebelumnya dengan data visualisasi lengkap...")
            df = pd.read_csv(HASIL_CSV_PATH, encoding="utf-8-sig")
            model = load_model(MODEL_PATH)
            with open(LABEL_ENCODER_PATH, "rb") as f:
                le = pickle.load(f)

            # Load data visualisasi dari cache
            architecture_results, training_histories, comparison_data, full_results = (
                load_comprehensive_cache()
            )

            if "Status" in df.columns and "Prediksi" in df.columns:
                y_actual = df["Status"].astype(str)
                y_pred = df["Prediksi"].astype(str)

                # Inisialisasi variabel sebelum try-catch
                labels = []
                macro_f1 = 0.0
                accuracy_val = 0.0
                precision_per_class = []
                recall_per_class = []
                f1_per_class = []
                cm = []
                inference_time_per_sample = 0.0
                class_report = {}

                # Hitung metrik evaluasi final dengan error handling - DIPERBAIKI
                try:
                    # Macro F1 (utama)
                    macro_f1 = (
                        f1_score(y_actual, y_pred, average="macro", zero_division=0)
                        * 100
                    )

                    # Precision, Recall per kelas
                    precision_per_class = (
                        precision_score(y_actual, y_pred, average=None, zero_division=0)
                        * 100
                    )
                    recall_per_class = (
                        recall_score(y_actual, y_pred, average=None, zero_division=0)
                        * 100
                    )
                    f1_per_class = (
                        f1_score(y_actual, y_pred, average=None, zero_division=0) * 100
                    )

                    # Accuracy (sekunder)
                    accuracy_val = accuracy_score(y_actual, y_pred) * 100

                    # Confusion matrix
                    labels = list(np.unique(y_actual))
                    cm = confusion_matrix(y_actual, y_pred, labels=labels)

                    # Inference time measurement
                    inference_start = time.time()
                    for _ in range(10):
                        model.predict(
                            df.drop(columns=["Status", "Prediksi"])
                            .iloc[:100]
                            .to_numpy(),
                            verbose=0,
                        )
                    inference_time = (time.time() - inference_start) / 10
                    inference_time_per_sample = (inference_time / 100) * 1000

                    # Classification report
                    class_report = classification_report(
                        y_actual, y_pred, output_dict=True, zero_division=0
                    )

                except Exception as e:
                    print(f"⚠️  Error in final metrics calculation: {e}")
                    # Gunakan nilai default yang sudah diinisialisasi

                df_display = df.head(1000)

                response_data = {
                    "message": "Data hasil sebelumnya berhasil dimuat dengan visualisasi lengkap.",
                    "train": df_display.to_dict(orient="records"),
                    # Metrik evaluasi utama
                    "macro_f1": safe_round(macro_f1),
                    "accuracy": safe_round(accuracy_val),
                    "precision_per_class": [safe_round(p) for p in precision_per_class]
                    if isinstance(precision_per_class, (list, np.ndarray))
                    else [],
                    "recall_per_class": [safe_round(r) for r in recall_per_class]
                    if isinstance(recall_per_class, (list, np.ndarray))
                    else [],
                    "f1_per_class": [safe_round(f) for f in f1_per_class]
                    if isinstance(f1_per_class, (list, np.ndarray))
                    else [],
                    "inference_time_ms": safe_round(inference_time_per_sample, 4),
                    "labels": labels,
                    "confusion": safe_confusion_matrix(cm),
                    "classification_report": class_report,
                    "total_records": len(df),
                    "displayed_records": len(df_display),
                    "cache_loaded": True,
                }

                # Tambahkan data visualisasi jika ada
                if architecture_results and training_histories and comparison_data:
                    response_data["architecture_results"] = architecture_results
                    response_data["training_histories"] = training_histories
                    response_data["architecture_comparison"] = comparison_data
                    response_data["has_visualization_data"] = True

                    # Tambahkan full_results info jika ada
                    if full_results:
                        response_data["full_results"] = full_results
                        response_data["cache_timestamp"] = full_results.get(
                            "timestamp", "Unknown"
                        )

                    # Tambahkan best architecture info jika ada
                    if architecture_results:
                        best_arch = max(
                            architecture_results,
                            key=lambda x: (
                                x.get("macro_f1", 0),  # Primary: Macro F1
                                x.get("val_accuracy", 0),  # Secondary: Accuracy
                            ),
                        )
                        response_data["best_architecture"] = best_arch["architecture"]
                        response_data["best_hyperparams"] = best_arch["hyperparams"]

                    print("✅ Data visualisasi berhasil dimuat ke response")
                else:
                    response_data["has_visualization_data"] = False
                    print("⚠️  Data visualisasi tidak lengkap")

                return jsonify(response_data)
            else:
                df_display = df.head(1000)
                response_data = {
                    "message": "Data hasil sebelumnya berhasil dimuat.",
                    "train": df_display.to_dict(orient="records"),
                    "total_records": len(df),
                    "displayed_records": len(df_display),
                    "has_visualization_data": False,
                    "cache_loaded": True,
                }
                return jsonify(response_data)

        except Exception as e:
            print(f"Error loading existing results: {str(e)}")
            # Continue dengan proses baru

    # Jika tidak ada hasil tersimpan atau cache tidak lengkap, lanjutkan dengan proses normal
    file = request.files.get("file")
    if not file or not file.filename or not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "File harus CSV."}), 400

    try:
        csv_path = os.path.join(ROOT_UPLOAD_FOLDER, file.filename)
        file.save(csv_path)

        X, y, encoder, output_dim, df_clean = load_and_prepare_data(csv_path)

        print(
            f"📁 Data loaded: {X.shape[0]} samples, {X.shape[1]} features, {output_dim} classes"
        )

        # Set seed global untuk reproducibility
        set_seeds(42)

        # Gunakan evaluasi arsitektur dengan hyperparameter tuning
        (
            model,
            best_architecture,
            best_hyperparams,
            architecture_results,
            training_histories,
        ) = analyze_architectures_with_tuning(X, y, output_dim)

        # PERBAIKAN KRITIS: Pastikan model tidak None sebelum digunakan
        if model is None:
            raise ValueError(
                "Model training gagal, model tidak terinisialisasi. Silakan coba lagi dengan data yang berbeda."
            )

        le = encoder

        # Simpan model (sekarang aman karena sudah divalidasi)
        model.save(MODEL_PATH)
        print("💾 Model berhasil disimpan")

        # Prediksi training set (sekarang aman karena sudah divalidasi)
        y_pred = le.inverse_transform(np.argmax(model.predict(X, verbose=0), axis=1))
        y_actual = le.inverse_transform(y)

        df_clean["Prediksi"] = y_pred
        df_clean.to_csv(HASIL_CSV_PATH, index=False, encoding="utf-8-sig")
        print("💾 Hasil prediksi berhasil disimpan")

        # Inisialisasi variabel sebelum try-catch
        labels = []
        macro_f1 = 0.0
        accuracy_val = 0.0
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        cm = []
        inference_time_per_sample = 0.0
        class_report = {}

        # Hitung metrik evaluasi final dengan error handling - DIPERBAIKI
        try:
            # Macro F1 (utama)
            macro_f1 = (
                f1_score(y_actual, y_pred, average="macro", zero_division=0) * 100
            )

            # Precision, Recall per kelas
            precision_per_class = (
                precision_score(y_actual, y_pred, average=None, zero_division=0) * 100
            )
            recall_per_class = (
                recall_score(y_actual, y_pred, average=None, zero_division=0) * 100
            )
            f1_per_class = (
                f1_score(y_actual, y_pred, average=None, zero_division=0) * 100
            )

            # Accuracy (sekunder)
            accuracy_val = accuracy_score(y_actual, y_pred) * 100

            # Confusion matrix
            labels = list(np.unique(y_actual))
            cm = confusion_matrix(y_actual, y_pred, labels=labels)

            # Inference time measurement
            inference_start = time.time()
            for _ in range(10):
                model.predict(X[:100], verbose=0)
            inference_time = (time.time() - inference_start) / 10
            inference_time_per_sample = (inference_time / 100) * 1000

            # Classification report
            class_report = classification_report(
                y_actual, y_pred, output_dict=True, zero_division=0
            )

        except Exception as e:
            print(f"⚠️  Error in final metrics calculation: {e}")
            # Gunakan nilai default yang sudah diinisialisasi

        df_display = df_clean.head(1000)

        # Pastikan architecture_results bisa di-serialisasi ke JSON
        serializable_results = []
        for result in architecture_results:
            serializable_result = result.copy()
            # Hapus key yang tidak bisa di-serialisasi jika ada
            if "model" in serializable_result:
                del serializable_result["model"]
            # Pastikan semua nilai dalam history adalah float
            if "history" in serializable_result:
                for key in serializable_result["history"]:
                    serializable_result["history"][key] = [
                        float(x) for x in serializable_result["history"][key]
                    ]
            serializable_results.append(serializable_result)

        # Siapkan data untuk comparison chart dan tabel
        comparison_data = prepare_architecture_comparison_data(serializable_results)

        # SIMPAN DATA VISUALISASI KE CACHE - DENGAN FULL_RESULTS
        save_comprehensive_cache(
            serializable_results, training_histories, comparison_data
        )

        # Siapkan data untuk response
        response_data = {
            "message": f"Pelatihan selesai dengan Hyperparameter Tuning. Arsitektur terbaik: {best_architecture}",
            "train": df_display.to_dict(orient="records"),
            # Metrik evaluasi utama
            "macro_f1": safe_round(macro_f1),
            "accuracy": safe_round(accuracy_val),
            "precision_per_class": [safe_round(p) for p in precision_per_class]
            if isinstance(precision_per_class, (list, np.ndarray))
            else [],
            "recall_per_class": [safe_round(r) for r in recall_per_class]
            if isinstance(recall_per_class, (list, np.ndarray))
            else [],
            "f1_per_class": [safe_round(f) for f in f1_per_class]
            if isinstance(f1_per_class, (list, np.ndarray))
            else [],
            "inference_time_ms": safe_round(inference_time_per_sample, 4),
            "labels": labels,
            "confusion": safe_confusion_matrix(cm),
            "classification_report": class_report,
            "total_records": len(df_clean),
            "displayed_records": len(df_display),
            "best_architecture": best_architecture,
            "best_hyperparams": best_hyperparams,
            "architecture_results": serializable_results,
            "training_histories": training_histories,
            "architecture_comparison": comparison_data,
            "has_visualization_data": True,
            "cache_saved": True,
        }

        print("✅ Proses selesai dengan sukses, data visualisasi tersimpan")
        return jsonify(response_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses: {str(e)}"}), 500


@perhitungan_bp.route("/load-result", methods=["GET"])
def load_result():
    """Memuat hasil perhitungan yang tersimpan"""
    if not os.path.exists(HASIL_CSV_PATH):
        return jsonify({"error": "Tidak ada data hasil yang tersimpan."}), 404

    try:
        df = pd.read_csv(HASIL_CSV_PATH, encoding="utf-8-sig")

        # Load data visualisasi dari cache
        architecture_results, training_histories, comparison_data, full_results = (
            load_comprehensive_cache()
        )

        # Inisialisasi variabel sebelum try-catch
        labels = []
        macro_f1 = 0.0
        accuracy_val = 0.0
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        cm = []
        inference_time_per_sample = 0.0
        class_report = {}

        if "Status" in df.columns and "Prediksi" in df.columns:
            y_actual = df["Status"].astype(str)
            y_pred = df["Prediksi"].astype(str)

            # Hitung metrik evaluasi final dengan error handling - DIPERBAIKI
            try:
                # Macro F1 (utama)
                macro_f1 = (
                    f1_score(y_actual, y_pred, average="macro", zero_division=0) * 100
                )

                # Precision, Recall per kelas
                precision_per_class = (
                    precision_score(y_actual, y_pred, average=None, zero_division=0)
                    * 100
                )
                recall_per_class = (
                    recall_score(y_actual, y_pred, average=None, zero_division=0) * 100
                )
                f1_per_class = (
                    f1_score(y_actual, y_pred, average=None, zero_division=0) * 100
                )

                # Accuracy (sekunder)
                accuracy_val = accuracy_score(y_actual, y_pred) * 100

                # Confusion matrix
                labels = list(np.unique(y_actual))
                cm = confusion_matrix(y_actual, y_pred, labels=labels)

                # Inference time measurement (jika model tersedia)
                if os.path.exists(MODEL_PATH):
                    model = load_model(MODEL_PATH)
                    X = df.drop(columns=["Status", "Prediksi"]).to_numpy()
                    inference_start = time.time()
                    for _ in range(10):
                        model.predict(X[:100], verbose=0)
                    inference_time = (time.time() - inference_start) / 10
                    inference_time_per_sample = (inference_time / 100) * 1000

                # Classification report
                class_report = classification_report(
                    y_actual, y_pred, output_dict=True, zero_division=0
                )

            except Exception as e:
                print(f"⚠️  Error in final metrics calculation: {e}")
                # Gunakan nilai default yang sudah diinisialisasi

            df_display = df.head(1000)

            response_data = {
                "train": df_display.to_dict(orient="records"),
                # Metrik evaluasi utama
                "macro_f1": safe_round(macro_f1),
                "accuracy": safe_round(accuracy_val),
                "precision_per_class": [safe_round(p) for p in precision_per_class]
                if isinstance(precision_per_class, (list, np.ndarray))
                else [],
                "recall_per_class": [safe_round(r) for r in recall_per_class]
                if isinstance(recall_per_class, (list, np.ndarray))
                else [],
                "f1_per_class": [safe_round(f) for f in f1_per_class]
                if isinstance(f1_per_class, (list, np.ndarray))
                else [],
                "inference_time_ms": safe_round(inference_time_per_sample, 4),
                "labels": labels,
                "confusion": safe_confusion_matrix(cm),
                "classification_report": class_report,
                "total_records": len(df),
                "displayed_records": len(df_display),
            }

            # Tambahkan data visualisasi jika ada
            if architecture_results and training_histories and comparison_data:
                response_data["architecture_results"] = architecture_results
                response_data["training_histories"] = training_histories
                response_data["architecture_comparison"] = comparison_data
                response_data["has_visualization_data"] = True
                response_data["message"] = "Data hasil dan visualisasi berhasil dimuat."

                # Tambahkan full_results info jika ada
                if full_results:
                    response_data["full_results"] = full_results
                    response_data["cache_timestamp"] = full_results.get(
                        "timestamp", "Unknown"
                    )

                # Tambahkan best architecture info jika ada - DENGAN KRITERIA BARU
                if architecture_results:
                    best_arch = max(
                        architecture_results,
                        key=lambda x: (
                            x.get("macro_f1", 0),  # Primary: Macro F1
                            x.get("val_accuracy", 0),  # Secondary: Accuracy
                        ),
                    )
                    response_data["best_architecture"] = best_arch["architecture"]
                    response_data["best_hyperparams"] = best_arch["hyperparams"]
            else:
                response_data["has_visualization_data"] = False
                response_data["message"] = (
                    "Data hasil berhasil dimuat, tetapi data visualisasi tidak tersedia."
                )

            return jsonify(response_data)
        else:
            df_display = df.head(1000)
            response_data = {
                "train": df_display.to_dict(orient="records"),
                "message": "Data hasil dimuat, tetapi tidak memiliki kolom yang diperlukan untuk evaluasi.",
                "total_records": len(df),
                "displayed_records": len(df_display),
                "has_visualization_data": False,
            }
            return jsonify(response_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memuat data hasil: {str(e)}"}), 500


@perhitungan_bp.route("/download/<filename>")
def download_file(filename):
    safe_files = {
        "model_mlp_custom.keras": MODEL_PATH,
        "label_encoder.pkl": LABEL_ENCODER_PATH,
        "hasil_perhitungan.csv": HASIL_CSV_PATH,
    }
    path = safe_files.get(filename)
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({"error": f"File '{filename}' tidak ditemukan."}), 404


@perhitungan_bp.route("/clear", methods=["POST"])
def clear_data():
    global model, le

    deleted_files = []

    # Hapus file utama
    for path in [MODEL_PATH, LABEL_ENCODER_PATH, HASIL_CSV_PATH]:
        if os.path.exists(path):
            os.remove(path)
            deleted_files.append(os.path.basename(path))

    # Hapus cache visualisasi
    success, cache_deleted = clear_visualization_cache()
    if success:
        deleted_files.extend(cache_deleted)

    model, le = None, None

    message = "Semua data dan model berhasil dihapus."
    if deleted_files:
        message = f"File berhasil dihapus: {', '.join(deleted_files)}"

    return jsonify({"message": message, "deleted_files": deleted_files})


@perhitungan_bp.route("/copy-download-model")
def copy_and_download_model():
    try:
        if os.path.exists(MODEL_PATH):
            return send_file(MODEL_PATH, as_attachment=True)
        else:
            return jsonify({"error": "Model tidak ditemukan."}), 404
    except Exception as e:
        return jsonify({"error": f"Gagal mendownload model: {str(e)}"}), 500


@perhitungan_bp.route("/copy-download-encoder")
def copy_and_download_encoder():
    try:
        if os.path.exists(LABEL_ENCODER_PATH):
            return send_file(LABEL_ENCODER_PATH, as_attachment=True)
        else:
            return jsonify({"error": "Label encoder tidak ditemukan."}), 404
    except Exception as e:
        return jsonify({"error": f"Gagal mendownload encoder: {str(e)}"}), 500


# Route untuk memeriksa status cache visualisasi - DIPERBAIKI
@perhitungan_bp.route("/check-visualization-cache", methods=["GET"])
def check_visualization_cache():
    """Memeriksa status cache visualisasi - VERSI DIPERBAIKI"""
    try:
        cache_complete, missing_files = validate_cache_completeness()
        cache_status = get_cache_status()

        # Debug information
        debug_cache_contents()

        # Format response yang lebih detail
        cache_files = {}
        for name, info in cache_status.items():
            cache_files[name] = info["exists"]

        return jsonify(
            {
                "has_complete_cache": cache_complete,
                "missing_files": missing_files,
                "cache_files": cache_files,
                "cache_details": cache_status,  # Kirim detail lengkap
            }
        )
    except Exception as e:
        print(f"❌ Error dalam check-visualization-cache: {e}")
        return jsonify({"has_complete_cache": False, "error": str(e)})


# Route untuk memuat hanya data visualisasi - DIPERBAIKI
@perhitungan_bp.route("/load-visualization-data", methods=["GET"])
def load_visualization_data():
    """Memuat hanya data visualisasi dari cache - VERSI DIPERBAIKI"""
    try:
        print("🔄 Memuat data visualisasi dari cache...")
        architecture_results, training_histories, comparison_data, full_results = (
            load_comprehensive_cache()
        )

        # Debug informasi
        print(f"Architecture results: {architecture_results is not None}")
        print(f"Training histories: {training_histories is not None}")
        print(f"Comparison data: {comparison_data is not None}")
        print(f"Full results: {full_results is not None}")

        if architecture_results and training_histories and comparison_data:
            response_data = {
                "architecture_results": architecture_results,
                "training_histories": training_histories,
                "architecture_comparison": comparison_data,
                "has_visualization_data": True,
                "message": "Data visualisasi berhasil dimuat dari cache.",
            }

            # Tambahkan full_results jika ada
            if full_results:
                response_data["full_results"] = full_results
                response_data["cache_timestamp"] = full_results.get(
                    "timestamp", "Unknown"
                )
                response_data["cache_version"] = full_results.get("version", "Unknown")
                print("✅ Full results termasuk dalam response")

            # Tambahkan best architecture info jika ada - DENGAN KRITERIA BARU
            if architecture_results:
                best_arch = max(
                    architecture_results,
                    key=lambda x: (
                        x.get("macro_f1", 0),  # Primary: Macro F1
                        x.get("val_accuracy", 0),  # Secondary: Accuracy
                    ),
                )
                response_data["best_architecture"] = best_arch.get(
                    "architecture", "Unknown"
                )
                response_data["best_hyperparams"] = best_arch.get("hyperparams", {})
                response_data["best_accuracy"] = best_arch.get("val_accuracy", 0)
                response_data["best_macro_f1"] = best_arch.get("macro_f1", 0)

            print("✅ Data visualisasi berhasil dimuat dan dikirim")
            return jsonify(response_data)
        else:
            print("❌ Data visualisasi tidak lengkap")
            missing = []
            if not architecture_results:
                missing.append("architecture_results")
            if not training_histories:
                missing.append("training_histories")
            if not comparison_data:
                missing.append("comparison_data")

            return jsonify(
                {
                    "has_visualization_data": False,
                    "message": f"Data visualisasi tidak lengkap. Missing: {', '.join(missing)}",
                    "missing_components": missing,
                }
            ), 404

    except Exception as e:
        print(f"❌ Gagal memuat data visualisasi: {str(e)}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"Gagal memuat data visualisasi: {str(e)}"}), 500


# Route untuk debug cache
@perhitungan_bp.route("/debug-cache", methods=["GET"])
def debug_cache():
    """Route untuk debugging cache"""
    try:
        cache_status = get_cache_status()
        debug_cache_contents()

        return jsonify(
            {
                "cache_status": cache_status,
                "debug_info": "Cache debug information printed to server console",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

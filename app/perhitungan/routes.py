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

# =============================================
# KONFIGURASI PATH DAN CACHE
# =============================================

# Folder utama
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROOT_UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "perhitungan")
os.makedirs(ROOT_UPLOAD_FOLDER, exist_ok=True)

# Jalur model & encoder
MODEL_PATH = os.path.join(ROOT_UPLOAD_FOLDER, "model_mlp_custom.keras")
LABEL_ENCODER_PATH = os.path.join(ROOT_UPLOAD_FOLDER, "label_encoder.pkl")
HASIL_CSV_PATH = os.path.join(ROOT_UPLOAD_FOLDER, "hasil_perhitungan.csv")

# JALUR CACHE UNTUK VISUALISASI - SISTEM KONSISTENSI UNIVERSAL
ARCHITECTURE_RESULTS_CACHE = os.path.join(
    ROOT_UPLOAD_FOLDER, "architecture_results_cache.pkl"
)
TRAINING_HISTORIES_CACHE = os.path.join(
    ROOT_UPLOAD_FOLDER, "training_histories_cache.pkl"
)
COMPARISON_DATA_CACHE = os.path.join(ROOT_UPLOAD_FOLDER, "comparison_data_cache.pkl")
FULL_RESULTS_CACHE = os.path.join(ROOT_UPLOAD_FOLDER, "full_results_cache.pkl")
BEST_RESULT_CACHE = os.path.join(ROOT_UPLOAD_FOLDER, "best_result_cache.pkl")

model = None
le = None


# =============================================
# FUNGSI UTILITAS DASAR
# =============================================


def safe_round(value, decimals=2):
    """Round value safely, handling None and NaN"""
    if value is None:
        return 0.0
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return 0.0


def safe_confusion_matrix(cm):
    """Convert confusion matrix to list safely"""
    if cm is None:
        return []
    if hasattr(cm, "tolist"):
        return cm.tolist()
    if isinstance(cm, (list, np.ndarray)):
        return cm
    return []


def set_seeds(seed=42):
    """Set fixed seeds untuk reproducibility"""
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
# SISTEM KONSISTENSI UNIVERSAL
# =============================================


def get_cache_status():
    """Mendapatkan status lengkap semua file cache"""
    cache_files = {
        "architecture_results": ARCHITECTURE_RESULTS_CACHE,
        "training_histories": TRAINING_HISTORIES_CACHE,
        "comparison_data": COMPARISON_DATA_CACHE,
        "full_results": FULL_RESULTS_CACHE,
        "best_result": BEST_RESULT_CACHE,
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
    print("\n" + "=" * 40)
    print("DEBUG CACHE CONTENTS")
    print("=" * 40)

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
                    f"\nArchitecture results: {len(arch_data) if arch_data else 0} items"
                )
                if arch_data:
                    for arch in arch_data:
                        print(
                            f"  - {arch.get('architecture', 'Unknown')}: "
                            f"Val Acc={arch.get('val_accuracy', 0)}%, "
                            f"Macro F1={arch.get('macro_f1', 0)}%"
                        )

        if os.path.exists(BEST_RESULT_CACHE):
            with open(BEST_RESULT_CACHE, "rb") as f:
                best_data = pickle.load(f)
                print(f"\nBest result: {best_data.get('architecture', 'Unknown')}")
                print(f"  Val Accuracy: {best_data.get('val_accuracy', 0)}%")
                print(f"  Macro F1: {best_data.get('macro_f1', 0)}%")

    except Exception as e:
        print(f"Error reading cache: {e}")

    print("=" * 40 + "\n")


def ensure_universal_consistency(architecture_results, best_result):
    """MEMASTIKAN data konsisten untuk SEMUA arsitektur terbaik"""
    if not architecture_results or not best_result:
        return architecture_results

    best_arch_name = best_result.get("architecture")
    print("\n🔧 SISTEM KONSISTENSI UNIVERSAL")
    print(f"   Target: {best_arch_name}")

    # Verifikasi bahwa best_result memiliki data lengkap
    required_fields = ["val_accuracy", "macro_f1", "macro_precision", "macro_recall"]
    for field in required_fields:
        if field not in best_result:
            print(f"   ⚠️  Best result tidak memiliki field {field}")
            best_result[field] = 0

    # Update data di architecture_results dengan best_result
    updated = False
    for i, arch_data in enumerate(architecture_results):
        if arch_data.get("architecture") == best_arch_name:
            print(f"   🔍 Memeriksa konsistensi untuk {best_arch_name}...")

            # Bandingkan semua field penting
            inconsistencies = []
            for field in required_fields:
                best_val = best_result.get(field, 0)
                arch_val = arch_data.get(field, 0)

                if abs(best_val - arch_val) > 0.01:  # Toleransi 0.01%
                    inconsistencies.append(
                        {
                            "field": field,
                            "best": best_val,
                            "current": arch_val,
                            "diff": abs(best_val - arch_val),
                        }
                    )

            if inconsistencies:
                print(f"   ⚠️  Ditemukan {len(inconsistencies)} ketidaksesuaian:")
                for inc in inconsistencies:
                    print(
                        f"      {inc['field']}: best={inc['best']}%, current={inc['current']}%"
                    )

                # PERBAIKAN UTAMA: Update SEMUA field dari best_result
                print(f"   🔄 Mengupdate data {best_arch_name}...")

                # Simpan beberapa field yang tidak ingin di-overwrite
                preserve_fields = [
                    "training_time_total",
                    "avg_memory_mb",
                    "inference_time_ms",
                    "total_params",
                    "epochs_used",
                    "success",
                ]
                preserved_data = {}
                for field in preserve_fields:
                    if field in arch_data:
                        preserved_data[field] = arch_data[field]

                # Update dengan data dari best_result
                architecture_results[i] = best_result.copy()

                # Restore field yang dipreserve
                for field, value in preserved_data.items():
                    architecture_results[i][field] = value

                print(f"   ✅ Data {best_arch_name} telah diperbarui")
                updated = True
            else:
                print(f"   ✅ Data {best_arch_name} sudah konsisten")

    # Jika arsitektur terbaik tidak ditemukan di architecture_results, tambahkan
    if not updated and best_arch_name:
        arch_found = any(
            arch.get("architecture") == best_arch_name for arch in architecture_results
        )
        if not arch_found:
            print(f"   ➕ Menambahkan {best_arch_name} ke architecture_results")
            architecture_results.append(best_result.copy())

    return architecture_results


def prepare_universal_comparison_data(architecture_results, best_result=None):
    """Mempersiapkan data untuk chart dan tabel perbandingan - VERSI UNIVERSAL"""
    if not architecture_results:
        return None

    # Pastikan architecture_results adalah list
    if isinstance(architecture_results, dict):
        architecture_results = list(architecture_results.values())

    # Urutkan berdasarkan urutan yang diinginkan: A, B, C, D
    desired_order = [
        "MLP-A (Sederhana)",
        "MLP-B (Sedang)",
        "MLP-C (Dalam)",
        "MLP-D (Wide + Shallow)",
    ]

    # Buat mapping untuk akses cepat
    result_mapping = {}
    for result in architecture_results:
        if isinstance(result, dict) and "architecture" in result:
            result_mapping[result["architecture"]] = result

    # GUNAKAN best_result sebagai SUMBER UTAMA untuk arsitektur terbaik
    if best_result and "architecture" in best_result:
        best_arch_name = best_result["architecture"]
        print(f"🎯 Menggunakan best_result sebagai sumber data untuk {best_arch_name}")

        # Update result_mapping dengan data dari best_result
        result_mapping[best_arch_name] = best_result.copy()

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

    # DEBUG: Tampilkan data untuk semua arsitektur
    print("\n📊 DATA UNTUK CHART/TABEL:")
    for result in ordered_results:
        arch_name = result["architecture"]
        if best_result and arch_name == best_result.get("architecture"):
            print(
                f"  ⭐ {arch_name}: Val Acc={result.get('val_accuracy', 0)}%, "
                f"Macro F1={result.get('macro_f1', 0)}% (BEST RESULT)"
            )
        else:
            print(
                f"  📋 {arch_name}: Val Acc={result.get('val_accuracy', 0)}%, "
                f"Macro F1={result.get('macro_f1', 0)}%"
            )

    # Siapkan data untuk chart
    comparison_data = {
        "labels": [result["architecture"] for result in ordered_results],
        "accuracy": [result.get("val_accuracy", 0) for result in ordered_results],
        "macro_f1": [result.get("macro_f1", 0) for result in ordered_results],
        "macro_precision": [
            result.get("macro_precision", 0) for result in ordered_results
        ],
        "macro_recall": [result.get("macro_recall", 0) for result in ordered_results],
        "training_time": [
            result.get("training_time_total", 0) for result in ordered_results
        ],
        "memory_usage": [result.get("avg_memory_mb", 0) for result in ordered_results],
        "inference_time": [
            result.get("inference_time_ms", 0) for result in ordered_results
        ],
        "parameters": [result.get("total_params", 0) for result in ordered_results],
        "epochs_used": [result.get("epochs_used", 0) for result in ordered_results],
        "best_architecture": best_result.get("architecture") if best_result else None,
        "best_accuracy": best_result.get("val_accuracy", 0) if best_result else 0,
        "best_macro_f1": best_result.get("macro_f1", 0) if best_result else 0,
        "best_macro_precision": best_result.get("macro_precision", 0)
        if best_result
        else 0,
        "best_macro_recall": best_result.get("macro_recall", 0) if best_result else 0,
    }

    return comparison_data


def verify_universal_consistency(architecture_results, best_result):
    """Verifikasi konsistensi universal antara semua data"""
    if not best_result or not architecture_results:
        print("⚠️  Tidak ada data untuk diverifikasi")
        return False

    best_arch_name = best_result.get("architecture")
    print(f"\n🔍 VERIFIKASI KONSISTENSI UNTUK {best_arch_name}:")

    # Cari data di architecture_results
    arch_data = None
    for result in architecture_results:
        if result.get("architecture") == best_arch_name:
            arch_data = result
            break

    if not arch_data:
        print(f"❌ {best_arch_name} tidak ditemukan di architecture_results")
        return False

    # Bandingkan semua field penting
    key_fields = [
        ("val_accuracy", "Validation Accuracy"),
        ("macro_f1", "Macro F1"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
    ]

    all_consistent = True
    for field_key, field_name in key_fields:
        best_val = best_result.get(field_key, 0)
        arch_val = arch_data.get(field_key, 0)

        if abs(best_val - arch_val) <= 0.01:  # Toleransi 0.01%
            print(f"   ✅ {field_name}: KONSISTEN ({best_val}%)")
        else:
            print(f"   ❌ {field_name}: TIDAK KONSISTEN!")
            print(f"      best_result: {best_val}%")
            print(f"      architecture_results: {arch_val}%")
            all_consistent = False

    if all_consistent:
        print(f"🎉 SEMUA DATA UNTUK {best_arch_name} KONSISTEN!")
    else:
        print(f"⚠️  ADA KETIDAKKONSISTENAN UNTUK {best_arch_name}")

    return all_consistent


def save_universal_cache(architecture_results, training_histories, best_result=None):
    """Menyimpan semua data dengan sistem konsistensi universal"""
    try:
        print("\n💾 MENYIMPAN CACHE DENGAN SISTEM KONSISTENSI UNIVERSAL...")

        # 1. PASTIKAN DATA KONSISTEN SEBELUM DISIMPAN
        if architecture_results and best_result:
            architecture_results = ensure_universal_consistency(
                architecture_results, best_result
            )

        # 2. BUAT COMPARISON DATA DARI DATA YANG SUDAH KONSISTEN
        comparison_data = prepare_universal_comparison_data(
            architecture_results, best_result
        )

        # 3. SIMPAN SEMUA DATA KE FILE TERPISAH
        # Architecture results (sudah konsisten)
        with open(ARCHITECTURE_RESULTS_CACHE, "wb") as f:
            pickle.dump(architecture_results, f)
        print(f"  ✅ architecture_results: {len(architecture_results)} items")

        # Training histories
        with open(TRAINING_HISTORIES_CACHE, "wb") as f:
            pickle.dump(training_histories, f)
        print("  ✅ training_histories: Disimpan")

        # Comparison data
        with open(COMPARISON_DATA_CACHE, "wb") as f:
            pickle.dump(comparison_data, f)
        print("  ✅ comparison_data: Disimpan")

        # Best result
        if best_result:
            with open(BEST_RESULT_CACHE, "wb") as f:
                pickle.dump(best_result, f)
            print(f"  ✅ best_result: Disimpan untuk {best_result.get('architecture')}")

        # 4. SIMPAN FULL RESULTS SEBAGAI SINGLE SOURCE OF TRUTH
        full_data = {
            "architecture_results": architecture_results,
            "training_histories": training_histories,
            "comparison_data": comparison_data,
            "best_result": best_result,
            "timestamp": datetime.now().isoformat(),
            "version": "4.0-universal",
            "total_architectures": len(architecture_results),
            "cache_info": "SINGLE SOURCE OF TRUTH - Data konsisten untuk semua arsitektur",
        }

        with open(FULL_RESULTS_CACHE, "wb") as f:
            pickle.dump(full_data, f)
        print("  ✅ full_results: SINGLE SOURCE OF TRUTH disimpan")

        # 5. VERIFIKASI KONSISTENSI SETELAH DISIMPAN
        verify_universal_consistency(architecture_results, best_result)

        print("✅ SEMUA DATA DISIMPAN DENGAN KONSISTENSI UNIVERSAL")
        return True, comparison_data
    except Exception as e:
        print(f"❌ Gagal menyimpan cache universal: {str(e)}")
        traceback.print_exc()
        return False, None


def load_universal_cache():
    """Memuat semua data dengan sistem konsistensi universal"""
    try:
        print("\n📂 MEMUAT CACHE DENGAN SISTEM KONSISTENSI UNIVERSAL...")

        # Coba muat dari FULL_RESULTS_CACHE sebagai single source of truth
        if os.path.exists(FULL_RESULTS_CACHE):
            with open(FULL_RESULTS_CACHE, "rb") as f:
                full_data = pickle.load(f)

            print("✅ Memuat dari SINGLE SOURCE OF TRUTH")

            architecture_results = full_data.get("architecture_results")
            training_histories = full_data.get("training_histories")
            comparison_data = full_data.get("comparison_data")
            best_result = full_data.get("best_result")

            # Verifikasi versi
            version = full_data.get("version", "unknown")
            print(f"📌 Versi cache: {version}")

        else:
            # Fallback: muat dari file individual
            print("⚠️  Single source of truth tidak ditemukan, memuat dari file individual")
            architecture_results = None
            training_histories = None
            comparison_data = None
            best_result = None

            if os.path.exists(ARCHITECTURE_RESULTS_CACHE):
                with open(ARCHITECTURE_RESULTS_CACHE, "rb") as f:
                    architecture_results = pickle.load(f)

            if os.path.exists(TRAINING_HISTORIES_CACHE):
                with open(TRAINING_HISTORIES_CACHE, "rb") as f:
                    training_histories = pickle.load(f)

            if os.path.exists(COMPARISON_DATA_CACHE):
                with open(COMPARISON_DATA_CACHE, "rb") as f:
                    comparison_data = pickle.load(f)

            if os.path.exists(BEST_RESULT_CACHE):
                with open(BEST_RESULT_CACHE, "rb") as f:
                    best_result = pickle.load(f)

            # Jika ada data, buat ulang comparison_data dengan konsistensi
            if architecture_results and best_result:
                architecture_results = ensure_universal_consistency(
                    architecture_results, best_result
                )
                comparison_data = prepare_universal_comparison_data(
                    architecture_results, best_result
                )

        # Verifikasi konsistensi setelah dimuat
        if architecture_results and best_result:
            verify_universal_consistency(architecture_results, best_result)

        print("✅ DATA DIMUAT DENGAN SISTEM KONSISTENSI UNIVERSAL")
        return architecture_results, training_histories, comparison_data, best_result

    except Exception as e:
        print(f"❌ Gagal memuat cache universal: {str(e)}")
        traceback.print_exc()
        return None, None, None, None


def validate_cache_completeness():
    """Validasi kelengkapan cache"""
    required_files = [
        FULL_RESULTS_CACHE,  # Utamakan single source of truth
        MODEL_PATH,
        LABEL_ENCODER_PATH,
        HASIL_CSV_PATH,
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(os.path.basename(file_path))

    return len(missing_files) == 0, missing_files


def clear_visualization_cache():
    """Menghapus cache visualisasi"""
    try:
        cache_files = [
            ARCHITECTURE_RESULTS_CACHE,
            TRAINING_HISTORIES_CACHE,
            COMPARISON_DATA_CACHE,
            FULL_RESULTS_CACHE,
            BEST_RESULT_CACHE,
        ]

        deleted_files = []
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                deleted_files.append(os.path.basename(cache_file))

        print("✅ Cache visualisasi berhasil dihapus")
        return True, deleted_files
    except Exception as e:
        print(f"❌ Gagal menghapus cache visualisasi: {str(e)}")
        return False, []


def ensure_list(value):
    """Memastikan nilai selalu berupa list untuk list comprehension"""
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [value]  # Konversi float tunggal ke list dengan 1 elemen
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    try:
        return list(value)  # Coba konversi ke list
    except:
        return []


# =============================================
# FUNGSI MLP ARCHITECTURES
# =============================================


def load_and_prepare_data(csv_path):
    """Memuat dan mempersiapkan data dari CSV"""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

        if "Status" not in df.columns:
            raise ValueError("Kolom 'Status' tidak ditemukan di CSV.")

        # Fitur
        X = df.drop(columns=["Status"]).to_numpy()

        # Label encoding
        y_raw = df["Status"].astype(str).to_numpy()
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_raw)

        if len(np.unique(y_encoded)) < 2:
            raise ValueError("Data harus memiliki minimal 2 kelas berbeda.")

        # Simpan encoder
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
        print(f"🧪 Mengevaluasi {architecture_name}")

        # Set seeds untuk reproducibility
        set_seeds(42)

        # Buat model
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

        # Training
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

        # Hitung epoch yang digunakan
        actual_epochs = len(history.history["accuracy"])
        time_per_epoch = training_time / actual_epochs if actual_epochs > 0 else 0

        # Hitung metrik
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

        # Prediksi untuk metrik
        y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
        y_true = y

        # Hitung semua metrik evaluasi menggunakan fungsi baru
        metrics = get_metrics_safely(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Inference time
        inference_start = time.time()
        for _ in range(10):
            model.predict(X[:100], verbose=0)
        inference_time = (time.time() - inference_start) / 10
        inference_time_per_sample = (inference_time / 100) * 1000

        # Classification report
        class_report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )

        # Hasil lengkap
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
            "macro_f1": safe_round(metrics["macro_f1"]),
            "macro_precision": safe_round(metrics["macro_precision"]),
            "macro_recall": safe_round(metrics["macro_recall"]),
            "accuracy": safe_round(metrics["accuracy"]),
            "precision_per_class": [safe_round(p) for p in metrics["precision_per_class"]],
            "recall_per_class": [safe_round(r) for r in metrics["recall_per_class"]],
            "f1_per_class": [safe_round(f) for f in metrics["f1_per_class"]],
            "inference_time_ms": safe_round(inference_time_per_sample, 4),
            "confusion_matrix": safe_confusion_matrix(cm),
            "classification_report": class_report,
            "success": True,
        }

        print(
            f"   ✅ {architecture_name}: "
            f"Val Acc={result['val_accuracy']}%, "
            f"Macro F1={result['macro_f1']}%"
        )

        # Clean up memory
        del history
        gc.collect()

        return result, model

    except Exception as e:
        print(
            f"   ❌ {architecture_name} dengan hyperparams {hyperparams} gagal: {str(e)}"
        )
        traceback.print_exc()
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
    print(f"🎯 Hyperparameter Tuning untuk {architecture_name}")

    # Generate semua kombinasi
    all_combinations = generate_hyperparameter_combinations()

    # Pilih kombinasi secara acak
    random.seed(42)
    selected_combinations = random.sample(
        all_combinations, min(max_combinations, len(all_combinations))
    )

    successful_models = []

    for i, hyperparams in enumerate(selected_combinations):
        print(f"   🔍 Kombinasi {i + 1}/{len(selected_combinations)}")

        result, model = evaluate_architecture_with_hyperparams(
            X, y, output_dim, architecture_func, architecture_name, hyperparams
        )

        if result.get("success", False) and model is not None:
            successful_models.append((result, model))

        time.sleep(0.5)

    if not successful_models:
        raise ValueError(
            f"Tidak ada kombinasi hyperparameter yang berhasil untuk {architecture_name}"
        )

    # Urutkan berdasarkan Macro F1 (kriteria utama)
    successful_models.sort(
        key=lambda x: (x[0]["macro_f1"], x[0]["val_accuracy"]), reverse=True
    )

    best_result, best_model = successful_models[0]

    print(f"✅ Hyperparameter terbaik untuk {architecture_name}:")
    print(f"   Macro F1: {best_result['macro_f1']}% (PRIMARY)")
    print(f"   Val Accuracy: {best_result['val_accuracy']}% (SECONDARY)")

    return (
        best_model,
        best_result["hyperparams"],
        [result for result, _ in successful_models],
    )


def analyze_architectures_with_tuning(X, y, output_dim):
    """Menganalisis semua arsitektur MLP dengan hyperparameter tuning - SISTEM UNIVERSAL"""
    # Definisikan urutan arsitektur yang FIXED
    architectures = [
        (create_mlp_a, "MLP-A (Sederhana)"),
        (create_mlp_b, "MLP-B (Sedang)"),
        (create_mlp_c, "MLP-C (Dalam)"),
        (create_mlp_d, "MLP-D (Wide + Shallow)"),
    ]

    all_architecture_results = []
    all_training_histories = {}
    all_successful_models = []

    print("\n" + "=" * 60)
    print("ANALISIS ARSITEKTUR MLP DENGAN SISTEM UNIVERSAL")
    print("=" * 60)

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
                all_architecture_results.append(best_arch_result)
                all_successful_models.append((best_arch_result, best_model, arch_name))

                # Simpan training history untuk chart
                all_training_histories[arch_name] = {
                    "accuracy": [best_arch_result.get("train_accuracy", 0) / 100],
                    "val_accuracy": [best_arch_result.get("val_accuracy", 0) / 100],
                    "loss": [0.5],
                    "val_loss": [0.5],
                }

                print(f"✅ {arch_name} berhasil ditambahkan")

            time.sleep(1)

        except Exception as e:
            print(f"❌ Arsitektur {arch_name} gagal dalam tuning: {str(e)}")
            # Tambahkan result kosong untuk menjaga urutan
            empty_result = {
                "architecture": arch_name,
                "val_accuracy": 0,
                "macro_f1": 0,
                "macro_precision": 0,
                "macro_recall": 0,
                "success": False,
                "error": str(e),
            }
            all_architecture_results.append(empty_result)
            continue

    if not all_successful_models:
        raise ValueError("Tidak ada arsitektur yang berhasil dilatih")

    # 🎯 SISTEM PEMILIHAN UNIVERSAL
    print("\n" + "=" * 60)
    print("SISTEM PEMILIHAN ARSITEKTUR TERBAIK")
    print("=" * 60)
    print("🎯 Kriteria UTAMA: Macro F1 Score (40%)")
    print("📊 Kriteria PENDUKUNG: Validation Accuracy (30%)")
    print("📈 Kriteria TAMBAHAN: Macro Precision (15%) & Recall (15%)")

    # Inisialisasi variabel untuk arsitektur terbaik
    best_overall_result = None
    best_overall_model = None
    best_overall_architecture = None
    best_overall_score = -1

    for result, model, arch_name in all_successful_models:
        # Hitung skor komposit dengan bobot
        macro_f1_weight = 0.4  # Bobot tertinggi
        val_acc_weight = 0.3
        macro_precision_weight = 0.15
        macro_recall_weight = 0.15

        composite_score = (
            result.get("macro_f1", 0) * macro_f1_weight
            + result.get("val_accuracy", 0) * val_acc_weight
            + result.get("macro_precision", 0) * macro_precision_weight
            + result.get("macro_recall", 0) * macro_recall_weight
        )

        print(f"\n🔍 {arch_name}:")
        print(f"   Macro F1: {result.get('macro_f1', 0)}% (bobot: {macro_f1_weight})")
        print(f"   Val Acc: {result.get('val_accuracy', 0)}% (bobot: {val_acc_weight})")
        print(
            f"   Macro Precision: {result.get('macro_precision', 0)}% (bobot: {macro_precision_weight})"
        )
        print(
            f"   Macro Recall: {result.get('macro_recall', 0)}% (bobot: {macro_recall_weight})"
        )
        print(f"   ⚖️  Skor Komposit: {composite_score:.2f}")

        if composite_score > best_overall_score:
            best_overall_score = composite_score
            best_overall_result = result
            best_overall_model = model
            best_overall_architecture = arch_name
            print(f"   🏆 LEADER BARU! Skor: {composite_score:.2f}")

    # PERBAIKAN KRITIS: Update data di all_architecture_results dengan best_overall_result
    if best_overall_result:
        print("\n" + "=" * 60)
        print(f"🏆 ARSITEKTUR TERBAIK DIPILIH: {best_overall_architecture}")
        print("=" * 60)
        print(f"   ⚖️  Skor Komposit: {best_overall_score:.2f}")
        print(f"   🎯 Macro F1: {best_overall_result.get('macro_f1', 0)}%")
        print(
            f"   ✅ Validation Accuracy: {best_overall_result.get('val_accuracy', 0)}%"
        )
        print(
            f"   📊 Macro Precision: {best_overall_result.get('macro_precision', 0)}%"
        )
        print(f"   📈 Macro Recall: {best_overall_result.get('macro_recall', 0)}%")

        # Update data di all_architecture_results
        for i, result in enumerate(all_architecture_results):
            if result.get("architecture") == best_overall_architecture:
                all_architecture_results[i] = best_overall_result.copy()
                print(
                    f"\n✅ Data {best_overall_architecture} di architecture_results telah diperbarui"
                )
                print(
                    f"   Val Accuracy: {best_overall_result.get('val_accuracy', 0)}% "
                    f"(sebelumnya: {result.get('val_accuracy', 0)}%)"
                )
                break
    else:
        raise ValueError("Tidak ada arsitektur yang memenuhi kriteria")

    return (
        best_overall_model,
        best_overall_architecture,
        best_overall_result["hyperparams"],
        all_architecture_results,
        all_training_histories,
        best_overall_result,  # best_result yang sudah konsisten
    )


# =============================================
# ROUTES DENGAN SISTEM KONSISTENSI UNIVERSAL
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

        # Cek cache
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
            print("📁 Memuat hasil sebelumnya dengan sistem universal...")
            df = pd.read_csv(HASIL_CSV_PATH, encoding="utf-8-sig")
            model = load_model(MODEL_PATH)
            with open(LABEL_ENCODER_PATH, "rb") as f:
                le = pickle.load(f)

            # Load data dengan sistem universal
            architecture_results, training_histories, comparison_data, best_result = (
                load_universal_cache()
            )

            if "Status" in df.columns and "Prediksi" in df.columns:
                y_actual = df["Status"].astype(str)
                y_pred = df["Prediksi"].astype(str)

                # Hitung metrik evaluasi final dengan penanganan yang aman
                macro_f1 = f1_score(y_actual, y_pred, average="macro", zero_division=0) * 100
                accuracy_val = accuracy_score(y_actual, y_pred) * 100
                
                # Hitung metrik per kelas - GUNAKAN average=None
                try:
                    precision_per_class = precision_score(y_actual, y_pred, average=None, zero_division=0) * 100
                    recall_per_class = recall_score(y_actual, y_pred, average=None, zero_division=0) * 100
                    f1_per_class = f1_score(y_actual, y_pred, average=None, zero_division=0) * 100
                    
                    # Konversi numpy array ke list jika perlu
                    if hasattr(precision_per_class, 'tolist'):
                        precision_per_class = precision_per_class.tolist()
                    if hasattr(recall_per_class, 'tolist'):
                        recall_per_class = recall_per_class.tolist()
                    if hasattr(f1_per_class, 'tolist'):
                        f1_per_class = f1_per_class.tolist()
                        
                except Exception as e:
                    print(f"⚠️  Error menghitung metrik per kelas: {e}")
                    precision_per_class = []
                    recall_per_class = []
                    f1_per_class = []
                
                # Pastikan selalu berupa list untuk list comprehension
                precision_per_class = ensure_list(precision_per_class)
                recall_per_class = ensure_list(recall_per_class)
                f1_per_class = ensure_list(f1_per_class)
                
                labels = list(np.unique(y_actual))
                cm = confusion_matrix(y_actual, y_pred, labels=labels)

                # Inference time
                inference_time_per_sample = 0
                if os.path.exists(MODEL_PATH):
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

                class_report = classification_report(
                    y_actual, y_pred, output_dict=True, zero_division=0
                )

                df_display = df.head(1000)

                response_data = {
                    "message": "Data hasil sebelumnya berhasil dimuat dengan sistem universal.",
                    "train": df_display.to_dict(orient="records"),
                    "macro_f1": safe_round(macro_f1),
                    "accuracy": safe_round(accuracy_val),
                    "precision_per_class": [safe_round(p) for p in precision_per_class],
                    "recall_per_class": [safe_round(r) for r in recall_per_class],
                    "f1_per_class": [safe_round(f) for f in f1_per_class],
                    "inference_time_ms": safe_round(inference_time_per_sample, 4),
                    "labels": labels,
                    "confusion": safe_confusion_matrix(cm),
                    "classification_report": class_report,
                    "total_records": len(df),
                    "displayed_records": len(df_display),
                    "cache_loaded": True,
                }

                # Tambahkan data visualisasi jika ada
                if (
                    architecture_results
                    and training_histories
                    and comparison_data
                    and best_result
                ):
                    response_data.update(
                        {
                            "architecture_results": architecture_results,
                            "training_histories": training_histories,
                            "architecture_comparison": comparison_data,
                            "has_visualization_data": True,
                            "best_architecture": best_result.get("architecture"),
                            "best_hyperparams": best_result.get("hyperparams", {}),
                            "best_architecture_result": best_result,
                            "best_accuracy": best_result.get("val_accuracy", 0),
                            "best_macro_f1": best_result.get("macro_f1", 0),
                            "best_macro_precision": best_result.get(
                                "macro_precision", 0
                            ),
                            "best_macro_recall": best_result.get("macro_recall", 0),
                        }
                    )

                    print("✅ Data visualisasi berhasil dimuat dengan sistem universal")

                return jsonify(response_data)
            else:
                df_display = df.head(1000)
                return jsonify(
                    {
                        "message": "Data hasil sebelumnya berhasil dimuat.",
                        "train": df_display.to_dict(orient="records"),
                        "total_records": len(df),
                        "displayed_records": len(df_display),
                        "has_visualization_data": False,
                        "cache_loaded": True,
                    }
                )

        except Exception as e:
            print(f"Error loading existing results: {str(e)}")
            # Continue dengan proses baru

    # Jika tidak ada hasil tersimpan, lanjutkan dengan proses normal
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

        # Set seed global
        set_seeds(42)

        # Analisis arsitektur dengan sistem universal
        (
            model,
            best_architecture,
            best_hyperparams,
            architecture_results,
            training_histories,
            best_result,
        ) = analyze_architectures_with_tuning(X, y, output_dim)

        # Validasi model
        if model is None:
            raise ValueError("Model training gagal.")

        le = encoder

        # Simpan model
        model.save(MODEL_PATH)
        print("💾 Model berhasil disimpan")

        # Prediksi training set
        y_pred = le.inverse_transform(np.argmax(model.predict(X, verbose=0), axis=1))
        y_actual = le.inverse_transform(y)

        df_clean["Prediksi"] = y_pred
        df_clean.to_csv(HASIL_CSV_PATH, index=False, encoding="utf-8-sig")
        print("💾 Hasil prediksi berhasil disimpan")

        # Hitung metrik evaluasi final menggunakan fungsi baru
        metrics = get_metrics_safely(y_actual, y_pred)
        
        # Pastikan metrik per kelas adalah list
        precision_per_class = ensure_list(metrics["precision_per_class"])
        recall_per_class = ensure_list(metrics["recall_per_class"])
        f1_per_class = ensure_list(metrics["f1_per_class"])
        
        labels = list(np.unique(y_actual))
        cm = confusion_matrix(y_actual, y_pred, labels=labels)

        # Inference time
        inference_start = time.time()
        for _ in range(10):
            model.predict(X[:100], verbose=0)
        inference_time = (time.time() - inference_start) / 10
        inference_time_per_sample = (inference_time / 100) * 1000

        class_report = classification_report(
            y_actual, y_pred, output_dict=True, zero_division=0
        )

        df_display = df_clean.head(1000)

        # Pastikan architecture_results bisa di-serialisasi
        serializable_results = []
        for result in architecture_results:
            serializable_result = result.copy()
            if "model" in serializable_result:
                del serializable_result["model"]
            serializable_results.append(serializable_result)

        # SIMPAN DATA DENGAN SISTEM UNIVERSAL
        cache_success, comparison_data = save_universal_cache(
            serializable_results,
            training_histories,
            best_result,
        )

        if not cache_success:
            print("⚠️  Gagal menyimpan cache, tetapi proses tetap dilanjutkan")

        # Siapkan response dengan data KONSISTEN UNIVERSAL
        response_data = {
            "message": f"Pelatihan selesai. Arsitektur terbaik: {best_architecture}",
            "train": df_display.to_dict(orient="records"),
            # Metrik evaluasi
            "macro_f1": safe_round(metrics["macro_f1"]),
            "accuracy": safe_round(metrics["accuracy"]),
            "precision_per_class": [safe_round(p) for p in precision_per_class],
            "recall_per_class": [safe_round(r) for r in recall_per_class],
            "f1_per_class": [safe_round(f) for f in f1_per_class],
            "inference_time_ms": safe_round(inference_time_per_sample, 4),
            "labels": labels,
            "confusion": safe_confusion_matrix(cm),
            "classification_report": class_report,
            "total_records": len(df_clean),
            "displayed_records": len(df_display),
            # Data visualisasi KONSISTEN UNIVERSAL
            "best_architecture": best_architecture,
            "best_hyperparams": best_hyperparams,
            "best_architecture_result": best_result,
            "best_accuracy": best_result.get("val_accuracy", 0),
            "best_macro_f1": best_result.get("macro_f1", 0),
            "best_macro_precision": best_result.get("macro_precision", 0),
            "best_macro_recall": best_result.get("macro_recall", 0),
            "architecture_results": serializable_results,
            "training_histories": training_histories,
            "architecture_comparison": comparison_data,
            "has_visualization_data": True,
            "cache_saved": cache_success,
        }

        print("\n" + "=" * 60)
        print("PROSES SELESAI DENGAN SISTEM KONSISTENSI UNIVERSAL")
        print("=" * 60)
        print(f"🏆 Arsitektur Terbaik: {best_architecture}")
        print(f"🎯 Macro F1: {best_result.get('macro_f1', 0)}%")
        print(f"✅ Val Accuracy: {best_result.get('val_accuracy', 0)}%")
        print(f"📊 Macro Precision: {best_result.get('macro_precision', 0)}%")
        print(f"📈 Macro Recall: {best_result.get('macro_recall', 0)}%")
        print("=" * 60)

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

        # Load data dengan sistem universal
        architecture_results, training_histories, comparison_data, best_result = (
            load_universal_cache()
        )

        # Inisialisasi variabel
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

            # Hitung metrik evaluasi final menggunakan fungsi baru
            try:
                metrics = get_metrics_safely(y_actual, y_pred)
                
                macro_f1 = metrics["macro_f1"]
                accuracy_val = metrics["accuracy"]
                precision_per_class = ensure_list(metrics["precision_per_class"])
                recall_per_class = ensure_list(metrics["recall_per_class"])
                f1_per_class = ensure_list(metrics["f1_per_class"])
                
                labels = list(np.unique(y_actual))
                cm = confusion_matrix(y_actual, y_pred, labels=labels)

                # Inference time
                if os.path.exists(MODEL_PATH):
                    model = load_model(MODEL_PATH)
                    X = df.drop(columns=["Status", "Prediksi"]).to_numpy()
                    inference_start = time.time()
                    for _ in range(10):
                        model.predict(X[:100], verbose=0)
                    inference_time = (time.time() - inference_start) / 10
                    inference_time_per_sample = (inference_time / 100) * 1000

                class_report = classification_report(
                    y_actual, y_pred, output_dict=True, zero_division=0
                )

            except Exception as e:
                print(f"⚠️  Error in final metrics calculation: {e}")

            df_display = df.head(1000)

            response_data = {
                "train": df_display.to_dict(orient="records"),
                "macro_f1": safe_round(macro_f1),
                "accuracy": safe_round(accuracy_val),
                "precision_per_class": [safe_round(p) for p in precision_per_class],
                "recall_per_class": [safe_round(r) for r in recall_per_class],
                "f1_per_class": [safe_round(f) for f in f1_per_class],
                "inference_time_ms": safe_round(inference_time_per_sample, 4),
                "labels": labels,
                "confusion": safe_confusion_matrix(cm),
                "classification_report": class_report,
                "total_records": len(df),
                "displayed_records": len(df_display),
            }

            # Tambahkan data visualisasi jika ada
            if (
                architecture_results
                and training_histories
                and comparison_data
                and best_result
            ):
                response_data.update(
                    {
                        "architecture_results": architecture_results,
                        "training_histories": training_histories,
                        "architecture_comparison": comparison_data,
                        "has_visualization_data": True,
                        "message": "Data hasil dan visualisasi berhasil dimuat dengan sistem universal.",
                        "best_architecture": best_result.get("architecture"),
                        "best_hyperparams": best_result.get("hyperparams", {}),
                        "best_architecture_result": best_result,
                        "best_accuracy": best_result.get("val_accuracy", 0),
                        "best_macro_f1": best_result.get("macro_f1", 0),
                        "best_macro_precision": best_result.get("macro_precision", 0),
                        "best_macro_recall": best_result.get("macro_recall", 0),
                    }
                )
            else:
                response_data.update(
                    {
                        "has_visualization_data": False,
                        "message": "Data hasil berhasil dimuat, tetapi data visualisasi tidak tersedia.",
                    }
                )

            return jsonify(response_data)
        else:
            df_display = df.head(1000)
            return jsonify(
                {
                    "train": df_display.to_dict(orient="records"),
                    "message": "Data hasil dimuat, tetapi tidak memiliki kolom yang diperlukan untuk evaluasi.",
                    "total_records": len(df),
                    "displayed_records": len(df_display),
                    "has_visualization_data": False,
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


@perhitungan_bp.route("/check-visualization-cache", methods=["GET"])
def check_visualization_cache():
    """Memeriksa status cache visualisasi"""
    try:
        cache_complete, missing_files = validate_cache_completeness()
        cache_status = get_cache_status()

        debug_cache_contents()

        cache_files = {}
        for name, info in cache_status.items():
            cache_files[name] = info["exists"]

        return jsonify(
            {
                "has_complete_cache": cache_complete,
                "missing_files": missing_files,
                "cache_files": cache_files,
                "cache_details": cache_status,
            }
        )
    except Exception as e:
        print(f"❌ Error dalam check-visualization-cache: {e}")
        return jsonify({"has_complete_cache": False, "error": str(e)})


@perhitungan_bp.route("/load-visualization-data", methods=["GET"])
def load_visualization_data():
    """Memuat hanya data visualisasi dengan sistem universal"""
    try:
        print("🔄 Memuat data visualisasi dengan sistem universal...")

        architecture_results, training_histories, comparison_data, best_result = (
            load_universal_cache()
        )

        if (
            architecture_results
            and training_histories
            and comparison_data
            and best_result
        ):
            response_data = {
                "architecture_results": architecture_results,
                "training_histories": training_histories,
                "architecture_comparison": comparison_data,
                "has_visualization_data": True,
                "message": "Data visualisasi berhasil dimuat dengan sistem universal.",
                "best_architecture": best_result.get("architecture"),
                "best_hyperparams": best_result.get("hyperparams", {}),
                "best_architecture_result": best_result,
                "best_accuracy": best_result.get("val_accuracy", 0),
                "best_macro_f1": best_result.get("macro_f1", 0),
                "best_macro_precision": best_result.get("macro_precision", 0),
                "best_macro_recall": best_result.get("macro_recall", 0),
            }

            print("✅ Data visualisasi berhasil dimuat dengan sistem universal")
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
            if not best_result:
                missing.append("best_result")

            return jsonify(
                {
                    "has_visualization_data": False,
                    "message": f"Data visualisasi tidak lengkap. Missing: {', '.join(missing)}",
                    "missing_components": missing,
                }
            ), 404

    except Exception as e:
        print(f"❌ Gagal memuat data visualisasi: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Gagal memuat data visualisasi: {str(e)}"}), 500


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


@perhitungan_bp.route("/validate-data-consistency", methods=["GET"])
def validate_data_consistency():
    """Validasi konsistensi data dengan sistem universal"""
    try:
        architecture_results, training_histories, comparison_data, best_result = (
            load_universal_cache()
        )

        if not architecture_results or not best_result:
            return jsonify(
                {"consistent": False, "message": "Data tidak lengkap untuk validasi"}
            )

        # Verifikasi konsistensi
        is_consistent = verify_universal_consistency(architecture_results, best_result)

        # Ambil data dari comparison_data untuk verifikasi lebih lanjut
        comparison_accuracy = (
            comparison_data.get("accuracy", []) if comparison_data else []
        )
        comparison_macro_f1 = (
            comparison_data.get("macro_f1", []) if comparison_data else []
        )

        # Cari indeks arsitektur terbaik di comparison_data
        best_arch_name = best_result.get("architecture")
        labels = comparison_data.get("labels", []) if comparison_data else []
        best_index = None
        for i, label in enumerate(labels):
            if label == best_arch_name:
                best_index = i
                break

        # Bandingkan data
        inconsistencies = []
        if best_index is not None:
            # Bandingkan dengan data di comparison_data
            comp_acc = (
                comparison_accuracy[best_index]
                if best_index < len(comparison_accuracy)
                else 0
            )
            comp_f1 = (
                comparison_macro_f1[best_index]
                if best_index < len(comparison_macro_f1)
                else 0
            )

            if abs(best_result.get("val_accuracy", 0) - comp_acc) > 0.1:
                inconsistencies.append(
                    {
                        "field": "Validation Accuracy",
                        "best_result": best_result.get("val_accuracy", 0),
                        "comparison_data": comp_acc,
                        "difference": abs(
                            best_result.get("val_accuracy", 0) - comp_acc
                        ),
                    }
                )

            if abs(best_result.get("macro_f1", 0) - comp_f1) > 0.1:
                inconsistencies.append(
                    {
                        "field": "Macro F1",
                        "best_result": best_result.get("macro_f1", 0),
                        "comparison_data": comp_f1,
                        "difference": abs(best_result.get("macro_f1", 0) - comp_f1),
                    }
                )

        return jsonify(
            {
                "consistent": is_consistent and len(inconsistencies) == 0,
                "best_architecture": best_result["architecture"],
                "best_accuracy": best_result.get("val_accuracy", 0),
                "best_macro_f1": best_result.get("macro_f1", 0),
                "inconsistencies": inconsistencies,
                "message": "✅ SEMUA DATA KONSISTEN (Sistem Universal)"
                if is_consistent and len(inconsistencies) == 0
                else "⚠️ Ada ketidakkonsistenan",
                "system_version": "universal_v4.0",
            }
        )

    except Exception as e:
        return jsonify({"consistent": False, "error": str(e)}), 500
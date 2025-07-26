import csv
import os
import re
import textwrap
import requests
from flask import Blueprint, request, jsonify, render_template, send_file

scraping_bp = Blueprint(
    "scraping",
    __name__,
    url_prefix="/scraping",
    template_folder="templates",
    static_folder="static",
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "scraping")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
SCRAPE_PATH = os.path.join(UPLOAD_FOLDER, "hasil_scraping.csv")


global_data_store = []  # simpan di luar fungsi, di level module


def get_data_store():
    return global_data_store


def wrap_text(text, width):
    return textwrap.fill(text, width)


def get_reviews(appid, params={"json": 1}):
    url = "https://store.steampowered.com/appreviews/"
    response = requests.get(
        url=url + appid, params=params, headers={"User-Agent": "Chrome/100.0.0.0"}
    )
    return response.json()


def get_n_reviews_balanced(appid, n=10):
    cursor = "*"
    days_in_5_years = 5 * 365 + 1
    positive_reviews = []
    negative_reviews = []
    max_attempts = 30  # batas maksimum loop
    attempt = 0

    params = {
        "json": 1,
        "filter": "all",
        "language": "english",
        "day_range": days_in_5_years,
        "review_type": "all",
        "purchase_type": "all",
    }

    target_positive = n // 2
    target_negative = n - target_positive

    while (
        len(positive_reviews) < target_positive
        or len(negative_reviews) < target_negative
    ) and attempt < max_attempts:
        attempt += 1
        params["cursor"] = cursor.encode()
        params["num_per_page"] = 100
        response = get_reviews(appid, params)
        cursor = response.get("cursor")
        fetched_reviews = response.get("reviews", [])

        if not fetched_reviews:
            print(f"[INFO] Tidak ada review baru pada iterasi ke-{attempt}.")
            break

        for review in fetched_reviews:
            raw_review = review.get("review", "")
            jumlah_kata = len(re.findall(r"\b\w+\b", raw_review))

            if jumlah_kata <= 20:
                continue

            if (
                "voted_up" in review
                and "author" in review
                and "steamid" in review["author"]
            ):
                if review["voted_up"] and len(positive_reviews) < target_positive:
                    positive_reviews.append(review)
                elif not review["voted_up"] and len(negative_reviews) < target_negative:
                    negative_reviews.append(review)

            if (
                len(positive_reviews) >= target_positive
                and len(negative_reviews) >= target_negative
            ):
                break

        print(
            f"[DEBUG] Iterasi ke-{attempt}: Positif = {len(positive_reviews)}, Negatif = {len(negative_reviews)}"
        )

        if len(fetched_reviews) < 100:
            print("[INFO] Jumlah review dari Steam sudah habis.")
            break

    total_valid = len(positive_reviews) + len(negative_reviews)
    if total_valid < n:
        print(
            f"[WARNING] Hanya {total_valid} review valid yang berhasil diambil dari {n} yang diminta."
        )

    return positive_reviews + negative_reviews


@scraping_bp.route("/")
def index():
    return render_template("data_scrap.html", page_name="scraping")


@scraping_bp.route("/scrapdat", methods=["POST"])
def scrapdat():
    data_store = get_data_store()
    appid = request.form.get("appid")
    num_reviews = int(request.form.get("num_reviews", 10))

    if not appid:
        return jsonify({"status": "error", "message": "App ID harus diisi."}), 400

    try:
        reviews = get_n_reviews_balanced(appid, num_reviews)
        processed_reviews = []

        seen_keys = set(r["id"] + r["review"] for r in data_store)

        for review in reviews:
            raw_review = review["review"]
            formatted_review = re.sub(r";", "", raw_review)
            unique_key = review["author"]["steamid"] + formatted_review

            if unique_key in seen_keys:
                continue

            seen_keys.add(unique_key)
            voted_up = "baik" if review["voted_up"] else "buruk"

            processed_reviews.append(
                {
                    "id": review["author"]["steamid"],
                    "review": formatted_review,
                    "status": voted_up,
                }
            )

            if len(processed_reviews) >= num_reviews:
                break

        data_store.extend(processed_reviews)

        os.makedirs(os.path.dirname(SCRAPE_PATH), exist_ok=True)
        with open(SCRAPE_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["SteamID", "Ulasan", "Status"])
            for review in data_store:
                formatted = review["review"].replace("\n", " ").replace("\r", " ")
                writer.writerow([review["id"], formatted, review["status"]])

        total_collected = len(processed_reviews)
        progress = min(int((total_collected / num_reviews) * 100), 100)

        return jsonify(
            {
                "status": "success",
                "data": processed_reviews,
                "total_data": len(data_store),
                "progress": progress,
                "positive_count": sum(1 for r in data_store if r["status"] == "baik"),
                "negative_count": sum(1 for r in data_store if r["status"] == "buruk"),
            }
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@scraping_bp.route("/getdata", methods=["GET"])
def getdata():
    data_store = get_data_store()
    print(
        f"[DEBUG] Total yang dikembalikan ke frontend: {len(data_store)}"
    )  # <--- pindahkan ke sini
    return jsonify(
        {
            "data": data_store,
            "total_data": len(data_store),
            "positive_count": sum(1 for r in data_store if r["status"] == "baik"),
            "negative_count": sum(1 for r in data_store if r["status"] == "buruk"),
        }
    )


@scraping_bp.route("/cleardata", methods=["POST"])
def cleardata():
    global_data_store.clear()

    return jsonify({"status": "success", "message": "Data berhasil dihapus."})


@scraping_bp.route("/savedata", methods=["GET"])
def savedata():
    data_store = get_data_store()

    if not data_store:
        return jsonify(
            {"status": "error", "message": "Tidak ada data untuk disimpan."}
        ), 400

    # Buat direktori jika belum ada
    os.makedirs(os.path.dirname(SCRAPE_PATH), exist_ok=True)

    try:
        # Tulis file ke disk
        with open(SCRAPE_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["SteamID", "Ulasan", "Status"])
            for review in data_store:
                formatted_review = (
                    review["review"].replace("\n", " ").replace("\r", " ")
                )
                writer.writerow([review["id"], formatted_review, review["status"]])
        
        if not os.path.isfile(SCRAPE_PATH):
            return jsonify({"error": "File belum tersedia. Silakan ambil data dulu."}), 404

        # Kirim file jika berhasil ditulis
        return send_file(
            SCRAPE_PATH,
            mimetype="text/csv",
            as_attachment=True,
            download_name="steam_reviews.csv",
        )

    except Exception as e:
        return jsonify({"error": f"Gagal menyimpan file: {str(e)}"}), 500


@scraping_bp.route("/download")
def download_csv():
    if os.path.exists(SCRAPE_PATH):
        return send_file(SCRAPE_PATH, as_attachment=True)
    return jsonify({"error": "File hasil scraping tidak ditemukan."}), 404

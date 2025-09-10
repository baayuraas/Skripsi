import csv
import os
import re
import textwrap
import time
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

global_data_store = []


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


def get_n_reviews_flexible(appid, n=10):
    target_positive = (n+1) // 2
    target_negative = n - target_positive

    def fetch_reviews_until(appid, target_count, is_positive):
        cursor = "*"
        results = []
        seen_review_ids = set()
        unchanged_cursor_count = 0

        while len(results) < target_count:
            params = {
                "json": 1,
                "filter": "all",
                "language": "english",
                "day_range": 1826,
                "review_type": "positive" if is_positive else "negative",
                "purchase_type": "all",
                "cursor": cursor.encode(),
                "num_per_page": 100,
            }

            response = get_reviews(appid, params)
            fetched_reviews = response.get("reviews", [])
            next_cursor = response.get("cursor", "*")

            if not fetched_reviews:
                break

            for review in fetched_reviews:
                raw_review = review.get("review", "")
                if len(re.findall(r"\b\w+\b", raw_review)) <= 20:
                    continue

                steamid = review.get("author", {}).get("steamid", "")
                review_id = review.get("recommendationid", "")

                if not steamid or not review_id:
                    continue

                unique_key = f"{steamid}_{review_id}"
                if unique_key in seen_review_ids:
                    continue

                seen_review_ids.add(unique_key)
                results.append(review)

                if len(results) >= target_count:
                    break

            print(
                f"[DEBUG] {('Positif' if is_positive else 'Negatif')}: {len(results)} / {target_count}"
            )

            if next_cursor == cursor:
                unchanged_cursor_count += 1
                if unchanged_cursor_count >= 3:
                    break
            else:
                unchanged_cursor_count = 0

            cursor = next_cursor
            time.sleep(0.5)

        return results

    positive_reviews = fetch_reviews_until(appid, target_positive, is_positive=True)
    negative_reviews = fetch_reviews_until(appid, target_negative, is_positive=False)

    if (
        len(positive_reviews) < target_positive
        or len(negative_reviews) < target_negative
    ):
        raise ValueError(
            f"Gagal mendapatkan jumlah review sesuai target. "
            f"Positif: {len(positive_reviews)}/{target_positive}, "
            f"Negatif: {len(negative_reviews)}/{target_negative}"
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

    if num_reviews < 2:
        return jsonify(
            {"status": "error", "message": "Jumlah review harus genap dan minimal 2."}
        ), 400

    if not appid:
        return jsonify({"status": "error", "message": "App ID harus diisi."}), 400

    try:
        reviews = get_n_reviews_flexible(appid, num_reviews)
        processed_reviews = []

        seen_keys = set(f"{r['id']}_{r['review']}" for r in data_store)

        for review in reviews:
            raw_review = review["review"]
            formatted_review = re.sub(r";", "", raw_review)
            unique_key = f"{review['author']['steamid']}_{formatted_review}"

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
    print(f"[DEBUG] Total yang dikembalikan ke frontend: {len(data_store)}")
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

    os.makedirs(os.path.dirname(SCRAPE_PATH), exist_ok=True)

    try:
        with open(SCRAPE_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["SteamID", "Ulasan", "Status"])
            for review in data_store:
                formatted_review = (
                    review["review"].replace("\n", " ").replace("\r", " ")
                )
                writer.writerow([review["id"], formatted_review, review["status"]])

        if not os.path.isfile(SCRAPE_PATH):
            return jsonify(
                {"error": "File belum tersedia. Silakan ambil data dulu."}
            ), 404

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

from flask import Flask, render_template


def create_app():
    app = Flask(__name__)

    # Register semua blueprint
    from .scraping.routes import scraping_bp
    from .terjemahan.routes import terjemahan_bp
    from .preproses.routes import prepro_bp, register_template_filters
    from .tfidf.routes import tfidf_bp
    from .perhitungan.routes import perhitungan_bp
    from .evaluasi.routes import evaluasi_bp

    app.register_blueprint(scraping_bp)
    app.register_blueprint(terjemahan_bp)
    app.register_blueprint(prepro_bp)
    app.register_blueprint(tfidf_bp)
    app.register_blueprint(perhitungan_bp)
    app.register_blueprint(evaluasi_bp)

    register_template_filters(app)

    # ✅ Tambahkan route root DI SINI:
    @app.route("/")
    def home():
        return render_template("index.html")

    return app

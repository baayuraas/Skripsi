# preprocessing_module.py
import os
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from functools import lru_cache
import logging
from collections import Counter

# Download NLTK resources
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except Exception as e:
    print(f"NLTK resources download failed: {e}")

class PreprocessingModule:
    """Modul preprocessing terpusat yang bisa digunakan oleh kedua program"""
    
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.txt_dir = os.path.join(self.base_dir, "app", "preproses")
        
        # Variabel state
        self._initialized = False
        self.stop_words_id = set()
        self.stop_words_ing = set()
        self.normalisasi_dict = {}
        self.stemming_dict = {}
        self.game_terms = set()
        self.kata_tidak_relevan = set()
        self.kata_id_pasti = set()
        
        # Pre-compile regex patterns
        self._compile_patterns()
        
        # Inisialisasi stemmer
        self.stemmer = StemmerFactory().create_stemmer()
        
        # Auto-initialize
        self.initialize_preprocessing_data()
    
    def _compile_patterns(self):
        """Compile semua regex patterns"""
        self.emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags (iOS)
            "\U00002702-\U000027b0"
            "\U000024c2-\U0001f251"
            "]+",
            flags=re.UNICODE,
        )
        
        self.special_char_pattern = re.compile(r"[-–—…\"»«]")
        self.bracket_pattern = re.compile(r"\[.*?\]")
        self.url_pattern = re.compile(r"http\S+")
        self.digit_pattern = re.compile(r"\b\d+\b")
        self.non_word_pattern = re.compile(r"[^\w\s@#]")
        self.whitespace_pattern = re.compile(r"\s+")
        self.repeated_word_pattern = re.compile(r"\b(\w{3,}?)(?:\1)\b")
        self.word_pattern = re.compile(r"\b\w+\b")
        self.sentence_split_pattern = re.compile(r"(?<=[.!?]) +|\n")
        
        # Pattern untuk membersihkan token tidak diinginkan
        self.number_unit_pattern = re.compile(r"^\d+[a-z]*\d*[a-z]*$")
        self.short_word_pattern = re.compile(r"^\w{1,2}$")
        self.mixed_alnum_pattern = re.compile(r"^[a-z]+\d+|\d+[a-z]+$")
    
    def load_stopwords(self, filename):
        """Memuat stopword dari file dan memastikan format konsisten"""
        stopwords_set = set()
        try:
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip().lower()
                    word = re.sub(r"[^a-z]", "", word)
                    if word and len(word) > 1:
                        stopwords_set.add(word)
            logging.info(f"Loaded {len(stopwords_set)} kata dari {filename}")
        except Exception as e:
            logging.error(f"Error load_stopwords {filename}: {e}")
        return stopwords_set
    
    def load_normalization_dict(self):
        """Fungsi loading yang robust untuk normalisasi_list.txt"""
        self.normalisasi_dict.clear()
        
        normalisasi_file = os.path.join(self.txt_dir, "normalisasi_list.txt")
        if not os.path.exists(normalisasi_file):
            logging.error("❌ File normalisasi_list.txt tidak ditemukan!")
            return
        
        loaded_count = 0
        error_count = 0
        
        try:
            with open(normalisasi_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip baris kosong dan komentar
                    if not line or line.startswith("#"):
                        continue
                    
                    # CARI POSISI "=" PERTAMA - lebih fleksibel
                    equal_pos = line.find("=")
                    if equal_pos == -1:
                        logging.warning(f"Baris {line_num}: Format tidak valid (tidak ada '='): {line}")
                        error_count += 1
                        continue
                    
                    # Split berdasarkan posisi =
                    k = line[:equal_pos].strip().lower()
                    v = line[equal_pos + 1:].strip().lower()
                    
                    # Validasi
                    if not k:
                        logging.warning(f"Baris {line_num}: Key kosong: {line}")
                        error_count += 1
                        continue
                    if not v:
                        logging.warning(f"Baris {line_num}: Value kosong: {line}")
                        error_count += 1
                        continue
                    
                    # Simpan ke dictionary
                    self.normalisasi_dict[k] = v
                    loaded_count += 1
                    
                    # Juga tambahkan variasi tanpa spasi jika ada spasi
                    if " " in k:
                        k_no_space = k.replace(" ", "")
                        if k_no_space != k and k_no_space not in self.normalisasi_dict:
                            self.normalisasi_dict[k_no_space] = v
                            loaded_count += 1
            
            logging.info(f"✅ SUCCESS: Loaded {loaded_count} entries dari normalisasi_list.txt, {error_count} errors")
            
        except Exception as e:
            logging.error(f"❌ Error load_normalization_dict: {e}")
    
    def initialize_preprocessing_data(self):
        """Fungsi untuk inisialisasi data preprocessing sekali saja"""
        if self._initialized:
            logging.debug("✅ Data preprocessing sudah diinisialisasi sebelumnya")
            return
        
        logging.info("🔄 Memulai inisialisasi data preprocessing...")
        
        # Daftar file txt yang diperlukan
        required_txt_files = [
            "stopword_list.txt", "stopword_list_ing.txt", "normalisasi_list.txt",
            "stemming_list.txt", "game_term.txt", "kata_tidak_relevan.txt", "kata_ambigu.txt"
        ]
        
        # Validasi file yang diperlukan
        for file_name in required_txt_files:
            file_path = os.path.join(self.txt_dir, file_name)
            if not os.path.exists(file_path):
                logging.warning(f"File {file_name} tidak ditemukan di {self.txt_dir}")
            else:
                logging.debug(f"File {file_name} ditemukan")
        
        # Inisialisasi stopword factory
        stopword_factory = StopWordRemoverFactory()
        
        # Load stopword dasar Indonesia
        self.stop_words_id = set(stopword_factory.get_stop_words()).union(
            set(stopwords.words("indonesian"))
        )
        
        # Load stopword Indonesia dari file dengan validasi
        id_stopwords = self.load_stopwords(os.path.join(self.txt_dir, "stopword_list.txt"))
        self.stop_words_id.update(id_stopwords)
        
        # Validasi stopword Indonesia
        expected_id_stopwords = {"yang", "dan", "di", "dari", "ke", "pada", "ini", "itu", "dengan", "untuk"}
        missing_id = expected_id_stopwords - self.stop_words_id
        if missing_id:
            logging.warning(f"Stopword ID yang hilang: {missing_id}")
            self.stop_words_id.update(missing_id)
        
        # Load stopword Inggris dari file
        stopword_ing_file = os.path.join(self.txt_dir, "stopword_list_ing.txt")
        if os.path.exists(stopword_ing_file):
            self.stop_words_ing = self.load_stopwords(stopword_ing_file)
        else:
            self.stop_words_ing = set(stopwords.words("english"))
            logging.info("Using NLTK English stopwords as fallback")
        
        # Validasi stopword Inggris
        expected_ing_stopwords = {"the", "and", "is", "in", "to", "of", "a", "for", "on", "with"}
        missing_ing = expected_ing_stopwords - self.stop_words_ing
        if missing_ing:
            logging.warning(f"Stopword ING yang hilang: {missing_ing}")
            self.stop_words_ing.update(missing_ing)
        
        logging.debug(f"Total stopwords Indonesia: {len(self.stop_words_id)}")
        logging.debug(f"Total stopwords Inggris: {len(self.stop_words_ing)}")
        
        # Load normalisasi dictionary dengan fungsi baru
        self.load_normalization_dict()
        
        # Load stemming_list.txt
        self.stemming_dict.clear()
        stemming_file = os.path.join(self.txt_dir, "stemming_list.txt")
        if os.path.exists(stemming_file):
            with open(stemming_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line:
                        parts = line.split("=", 1)
                        k = parts[0].strip().lower()
                        v = parts[1].strip().lower()
                        self.stemming_dict[k] = v
                        
                        k_normalized = re.sub(r"(.)\1+", r"\1", k)
                        if k_normalized != k and k_normalized not in self.stemming_dict:
                            self.stemming_dict[k_normalized] = v
            logging.debug(f"Loaded {len(self.stemming_dict)} pasangan dari stemming_list.txt")
        else:
            logging.warning("File stemming_list.txt tidak ditemukan, gunakan default Sastrawi.")
        
        # Load game terms
        self.game_terms.clear()
        game_terms_file = os.path.join(self.txt_dir, "game_term.txt")
        if os.path.exists(game_terms_file):
            with open(game_terms_file, "r", encoding="utf-8") as f:
                for line in f:
                    term = line.strip().lower()
                    if term:
                        self.game_terms.add(term)
            logging.debug(f"Loaded {len(self.game_terms)} game terms")
        else:
            logging.warning("File game_term.txt tidak ditemukan")
        
        # Load kata tidak relevan
        self.kata_tidak_relevan.clear()
        tidak_relevan_file = os.path.join(self.txt_dir, "kata_tidak_relevan.txt")
        if os.path.exists(tidak_relevan_file):
            with open(tidak_relevan_file, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        self.kata_tidak_relevan.add(word)
            logging.debug(f"Loaded {len(self.kata_tidak_relevan)} kata tidak relevan")
        else:
            logging.warning("File kata_tidak_relevan.txt tidak ditemukan")
        
        # Load kata ambigu (whitelist)
        self.kata_id_pasti.clear()
        whitelist_file = os.path.join(self.txt_dir, "kata_ambigu.txt")
        if os.path.exists(whitelist_file):
            try:
                with open(whitelist_file, "r", encoding="utf-8") as f:
                    self.kata_id_pasti = {line.strip().lower() for line in f if line.strip()}
                logging.debug(f"Loaded {len(self.kata_id_pasti)} kata dari kata_ambigu.txt")
            except Exception as e:
                logging.error(f"Error baca whitelist: {e}")
        else:
            logging.warning("File kata_ambigu.txt tidak ditemukan, whitelist kosong.")
        
        self._initialized = True
        logging.info("✅ Inisialisasi data preprocessing selesai")
    
    def normalize_repeated_letters(self, word):
        """Mengurangi pengulangan huruf yang berlebihan menjadi maksimal 1 huruf"""
        if len(word) <= 2:
            return word
        normalized = re.sub(r"(.)\1+", r"\1", word)
        return normalized
    
    def bersihkan_terjemahan(self, teks: str) -> str:
        """Fungsi bersihkan terjemahan"""
        if pd.isna(teks) or not isinstance(teks, str):
            return ""
        
        teks_asli = teks
        try:
            teks = self.emoji_pattern.sub(" ", teks)
            teks = self.special_char_pattern.sub(" ", teks)
            teks = self.bracket_pattern.sub(" ", teks)
            teks = self.url_pattern.sub(" ", teks)
            teks = self.digit_pattern.sub(" ", teks)
            teks = self.non_word_pattern.sub(" ", teks)
            teks = self.whitespace_pattern.sub(" ", teks).strip()
            
            if not teks.strip():
                teks = self.emoji_pattern.sub(" ", teks_asli)
                teks = self.url_pattern.sub(" ", teks)
                teks = self.whitespace_pattern.sub(" ", teks).strip()
            
            return teks
        except Exception as e:
            logging.error(f"Error bersihkan_terjemahan: {e}")
            return self.whitespace_pattern.sub(" ", teks_asli).strip()
    
    def normalisasi_teks(self, words, debug=False):
        """Fungsi normalisasi yang robust dengan multiple fallback"""
        hasil = []
        normalisasi_count = 0
        skipped_words = []
        
        for w in words:
            wl = w.lower().strip()
            
            # Skip jika kata kosong
            if not wl:
                continue
            
            # CLEANING LEBIH AGGRESIF - hapus karakter non-alphabet di awal/akhir
            wl_clean = re.sub(r"^[^a-z]+|[^a-z]+$", "", wl)
            
            # PRIORITAS 1: Cek kata bersih (setelah cleaning)
            if wl_clean and wl_clean in self.normalisasi_dict:
                mapped = self.normalisasi_dict[wl_clean]
                if debug:
                    logging.debug(f"🔧 Normalisasi (clean): '{wl}' -> '{wl_clean}' -> '{mapped}'")
                hasil.extend(mapped.split())
                normalisasi_count += 1
                continue
            
            # PRIORITAS 2: Cek kata asli (tanpa cleaning)
            if wl in self.normalisasi_dict:
                mapped = self.normalisasi_dict[wl]
                if debug:
                    logging.debug(f"🔧 Normalisasi (original): '{wl}' -> '{mapped}'")
                hasil.extend(mapped.split())
                normalisasi_count += 1
                continue
            
            # PRIORITAS 3: Cek variasi dengan menghilangkan pengulangan huruf
            wl_normalized = self.normalize_repeated_letters(wl)
            if wl_normalized != wl and wl_normalized in self.normalisasi_dict:
                mapped = self.normalisasi_dict[wl_normalized]
                if debug:
                    logging.debug(f"🔧 Normalisasi (repeated): '{wl}' -> '{wl_normalized}' -> '{mapped}'")
                hasil.extend(mapped.split())
                normalisasi_count += 1
                continue
            
            # Jika tidak ada di dictionary, pertahankan kata asli
            hasil.append(wl)
            skipped_words.append(wl)
        
        if debug:
            if normalisasi_count > 0:
                logging.info(f"✅ Dilakukan {normalisasi_count} normalisasi")
            if skipped_words:
                logging.info(f"❌ Kata tidak ternormalisasi: {skipped_words}")
        
        return hasil
    
    def bersihkan_token(self, tokens, debug=False):
        """Membersihkan token-token yang tidak diinginkan"""
        hasil = []
        skipped_count = 0
        
        for token in tokens:
            normalized_token = self.normalize_repeated_letters(token)
            
            # PRIORITAS: JANGAN hapus token yang ada di normalisasi_dict
            if normalized_token in self.normalisasi_dict:
                if debug:
                    logging.debug(f"🔧 Pertahankan token untuk normalisasi: '{normalized_token}'")
                hasil.append(normalized_token)
                continue
            
            # JANGAN hapus token yang merupakan hasil normalisasi
            is_normalized_result = any(
                normalized_token in value.split() for value in self.normalisasi_dict.values()
            )
            if is_normalized_result:
                if debug:
                    logging.debug(f"🔧 Pertahankan hasil normalisasi: '{normalized_token}'")
                hasil.append(normalized_token)
                continue
            
            # Filter token yang tidak diinginkan
            if (
                self.number_unit_pattern.match(normalized_token)
                or self.short_word_pattern.match(normalized_token)
                or self.mixed_alnum_pattern.match(normalized_token)
            ):
                if debug:
                    logging.debug(f"🗑️ Hapus token tidak diinginkan: '{normalized_token}'")
                skipped_count += 1
                continue
            
            if normalized_token.isdigit():
                if debug:
                    logging.debug(f"🗑️ Hapus token angka: '{normalized_token}'")
                skipped_count += 1
                continue
            
            hasil.append(normalized_token)
        
        if debug and skipped_count > 0:
            logging.info(f"🗑️ Dihapus {skipped_count} token tidak diinginkan")
        
        return hasil
    
    def hapus_stopword(self, words, debug=False):
        """Fungsi stopword removal"""
        if not words:
            return []
        
        if debug:
            logging.debug(f"Kata sebelum filter stopword: {words}")
        
        result = []
        stopword_removed = []
        
        for w in words:
            w_lower = w.lower()
            
            # Pengecualian 1: Game terms
            if w_lower in self.game_terms:
                if debug:
                    logging.debug(f"🎮 Menyimpan game term: '{w}'")
                result.append(w)
                continue
            
            # Pengecualian 2: Whitelist
            if w_lower in self.kata_id_pasti:
                if debug:
                    logging.debug(f"📝 Menyimpan kata whitelist: '{w}'")
                result.append(w)
                continue
            
            # Prioritas 1: Kata tidak relevan
            if w_lower in self.kata_tidak_relevan:
                if debug:
                    logging.debug(f"🗑️ Menghapus kata tidak relevan: '{w}'")
                stopword_removed.append(w)
                continue
            
            # Prioritas 2: Stopword Indonesia
            if w_lower in self.stop_words_id:
                if debug:
                    logging.debug(f"🗑️ Menghapus stopword Indonesia: '{w}'")
                stopword_removed.append(w)
                continue
            
            # Prioritas 3: Stopword Inggris
            if w_lower in self.stop_words_ing:
                if debug:
                    logging.debug(f"🗑️ Menghapus stopword Inggris: '{w}'")
                stopword_removed.append(w)
                continue
            
            # Jika lolos semua filter, simpan kata
            if debug:
                logging.debug(f"💾 Menyimpan kata: '{w}'")
            result.append(w)
        
        if debug:
            logging.debug(f"Kata yang dihapus: {stopword_removed}")
            logging.debug(f"Kata setelah filter stopword: {result}")
        
        return result
    
    @lru_cache(maxsize=10000)
    def cached_stemmer_stem(self, word):
        """Cache untuk stemming"""
        return self.stemmer.stem(word)
    
    def stemming_teks(self, words, debug=False):
        """Fungsi stemming"""
        hasil = []
        stem_methods = []
        
        for w in words:
            wl = w.lower()
            wl_normalized = self.normalize_repeated_letters(wl)
            method_used = "original"
            
            if wl in self.stemming_dict:
                mapped = self.stemming_dict[wl]
                hasil.extend(mapped.split())
                method_used = "custom_dict"
            elif wl_normalized in self.stemming_dict:
                mapped = self.stemming_dict[wl_normalized]
                hasil.extend(mapped.split())
                method_used = "custom_dict_normalized"
            else:
                stemmed_sastrawi = self.cached_stemmer_stem(wl_normalized)
                if stemmed_sastrawi != wl_normalized:
                    hasil.append(stemmed_sastrawi)
                    method_used = "sastrawi"
                else:
                    hasil.append(wl_normalized)
                    method_used = "normalized"
            
            stem_methods.append(method_used)
        
        if debug and stem_methods:
            method_counts = Counter(stem_methods)
            logging.debug(f"Stemming methods digunakan: {dict(method_counts)}")
        
        return hasil
    
    def proses_preprocessing_standar(self, teks, debug=False):
        """Fungsi preprocessing standar utama - SAMA PERSIS dengan preproses"""
        if pd.isna(teks) or not isinstance(teks, str) or not teks.strip():
            return ["", "", [], [], [], [], ""]  # 7 elemen
        
        # Bersihkan teks
        clean = self.bersihkan_terjemahan(teks)
        folded = clean.lower()
        
        if debug:
            logging.info(f"📥 Input: '{teks}'")
            logging.info(f"🧹 Clean: '{clean}'")
            logging.info(f"🔠 Case Folding: '{folded}'")
        
        # Tokenisasi
        try:
            token = word_tokenize(folded)
        except Exception:
            token = self.word_pattern.findall(folded)
        
        if debug:
            logging.info(f"🔪 Tokenisasi: {token}")
        
        # PERBAIKAN URUTAN: Normalisasi dilakukan SEBELUM pembersihan token
        norm = self.normalisasi_teks(token, debug) if token else []
        
        if debug:
            logging.info(f"🔧 Setelah normalisasi: {norm}")
        
        # Bersihkan token tidak diinginkan (setelah normalisasi)
        token_cleaned = self.bersihkan_token(norm, debug) if norm else []
        
        if debug:
            logging.info(f"🧽 Setelah bersihkan_token: {token_cleaned}")
        
        # Stopword removal
        stop = self.hapus_stopword(token_cleaned, debug) if token_cleaned else []
        
        if debug:
            logging.info(f"🚫 Setelah stopword removal: {stop}")
        
        # Stemming
        stem = self.stemming_teks(stop, debug) if stop else []
        
        if debug:
            logging.info(f"✂️ Setelah stemming: {stem}")
        
        # Gabungkan hasil
        hasil = " ".join(stem) if stem else folded
        
        if debug:
            logging.info(f"🎯 Hasil akhir: '{hasil}'")
        
        return [clean, folded, token, stop, norm, stem, hasil]
    
    def proses_baris_aman(self, text: str) -> list:
        """
        Preprocess text untuk penggunaan di pengujian
        Returns a list containing the preprocessed text.
        """
        try:
            # Gunakan fungsi preprocessing standar
            hasil_prepro = self.proses_preprocessing_standar(text, debug=False)
            
            # Ambil hasil akhir (index 6) dan kembalikan dalam list
            hasil_akhir = hasil_prepro[6] if hasil_prepro[6] else hasil_prepro[1]  # fallback ke case folding jika hasil kosong
            
            return [hasil_akhir]
        except Exception as e:
            logging.error(f"Preprocessing failed: {str(e)}")
            raise Exception(f"Text preprocessing failed: {str(e)}")

# Buat instance global untuk mudah diimport
preprocessor = PreprocessingModule()
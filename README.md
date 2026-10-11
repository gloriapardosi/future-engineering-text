# Feature Engineering Teks: Review Aplikasi MyTelkomsel

Exercise 5 mata kuliah **[ISI NAMA MATA KULIAH]**.
Repositori ini berisi penerapan berbagai teknik *feature engineering* pada teks review aplikasi MyTelkomsel yang sudah melalui tahap preprocessing di exercise sebelumnya, beserta analisis hasilnya.

## Anggota Kelompok

| Nama | NIM |
|---|---|
| Nadhila Alya Azzahra | 051 |
| Gloria Jesika Oktoniar Pardosi | 148 |

## Tujuan

Mengubah teks review yang sudah bersih (`Teks_Bersih`) menjadi representasi numerik (vektor) dengan beberapa metode, lalu membandingkan karakteristik dan hasilnya.

## Dataset

| Keterangan | Nilai |
|---|---|
| Sumber | Review aplikasi MyTelkomsel (hasil scraping dan preprocessing pada Exercise 4) |
| File | `review_mytelkomsel_bersih.csv` |
| Jumlah data awal | 2.000 review |
| Setelah membuang review kosong | 1.886 review |
| Ukuran vocabulary | 1.356 kata (BoW/TF-IDF), 1.379 kata (Word2Vec/FastText) |
| Kolom utama | `Rating`, `Teks_Bersih` |

Review yang kosong setelah preprocessing (misalnya hanya berisi emoji) dibuang karena tidak memiliki fitur.

## Metode yang Digunakan

**Representasi berbasis hitungan kata (sparse)**
1. **One Hot Encoding**: nilai 1 jika kata muncul di review, 0 jika tidak (`CountVectorizer(binary=True)`).
2. **Bag-of-Words**: jumlah kemunculan tiap kata di review (`CountVectorizer`).
3. **TF-IDF**: bobot kata berdasarkan frekuensi di review dan kelangkaannya di seluruh review (`TfidfVectorizer`).

**Representasi berbasis embedding (dense)**

4. **Word2Vec**: CBOW, `vector_size=100`, `window=5`, `min_count=1` (`gensim`).
5. **FastText**: Skip-gram, `vector_size=100`, `window=5`, `min_n=3`, `max_n=6`, `epochs=10` (`gensim`).

Untuk Word2Vec dan FastText, vektor satu review diperoleh dari rata-rata vektor seluruh kata di dalamnya. Hasil embedding divisualisasikan dengan UMAP dan Plotly.

## Struktur Repositori

```
.
├── FEATURE_ENGINEERING.ipynb          # notebook utama
├── review_mytelkomsel_bersih.csv      # dataset hasil preprocessing
└── README.md
```

## Cara Menjalankan

**Di Google Colab (disarankan)**
1. Upload `FEATURE_ENGINEERING.ipynb` dan `review_mytelkomsel_bersih.csv` ke Colab.
2. Jalankan semua sel dari atas ke bawah (*Runtime → Run all*).

**Di komputer sendiri**
```bash
pip install pandas numpy matplotlib scikit-learn nltk tqdm gensim umap-learn plotly
jupyter notebook FEATURE_ENGINEERING.ipynb
```

## Hasil Ringkas

Perbandingan karakteristik:

| Metode | Jenis | Shape | Tangkap makna antar kata | Tangani kata baru (typo) |
|---|---|---|---|---|
| One Hot | Sparse | 1886 × 1356 | Tidak | Tidak |
| Bag-of-Words | Sparse | 1886 × 1356 | Tidak | Tidak |
| TF-IDF | Sparse | 1886 × 1356 | Tidak | Tidak |
| Word2Vec | Dense | 1886 × 100 | Ya | Tidak |
| FastText | Dense | 1886 × 100 | Ya | Ya |

Evaluasi sederhana untuk membandingkan kualitas fitur: klasifikasi review negatif (rating 1-2) vs positif (rating 4-5) memakai Logistic Regression dan 5-fold cross-validation.

| Fitur | Accuracy | F1 macro |
|---|---|---|
| TF-IDF | 0,899 | 0,861 |
| One Hot | 0,857 | 0,821 |
| Bag-of-Words | 0,844 | 0,808 |
| FastText | ± 0,75 | ± 0,68 |
| Word2Vec | ± 0,68 | ± 0,62 |

> Angka Word2Vec dan FastText dapat sedikit berbeda setiap dijalankan karena adanya unsur acak pada proses training.

## Temuan Utama

- **TF-IDF memberi hasil terbaik** pada data ini. Kata khas tiap kelompok terlihat jelas: review rating 1-2 didominasi kata seperti `harga`, `lot`, `jelek`, `buruk`, sedangkan rating 4-5 didominasi `bagus`, `mantap`, `baik`, `puas`.
- **Word2Vec dan FastText belum optimal** karena data kecil (sekitar 1.900 review pendek) dan jumlah epoch default rendah. Skor kemiripan kata hampir seragam (~0,99), sehingga kata-kata terdekat belum bermakna.
- **FastText unggul pada kata typo/slang** seperti `lemotttt` dan `mahaaal`, yang tidak bisa diberi vektor oleh Word2Vec (`KeyError`), padahal review pengguna penuh typo.
- **Data tidak seimbang**: review rating 1 sangat dominan, sehingga F1 macro lebih adil dibanding accuracy.
- Teknik berbasis hitungan kata lebih efektif untuk data kecil, sedangkan embedding membutuhkan data lebih besar atau model pretrained untuk menangkap makna dengan baik.

Analisis lengkap ada di bagian akhir notebook.

## Teknologi

Python · pandas · NumPy · scikit-learn · NLTK · gensim · UMAP · Plotly · Matplotlib

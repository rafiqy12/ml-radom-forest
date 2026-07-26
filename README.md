# Klasifikasi Forest Cover Type

Aplikasi Streamlit untuk klasifikasi tipe tutupan hutan menggunakan Random
Forest dan seleksi fitur berbasis Decision Tree (pendekatan C5.0).

## Menjalankan secara lokal

```bash
pip install -r requirements.txt
streamlit run mls2.py
```

Dataset Kaggle `uciml/forest-cover-type-dataset` sudah disimpan sebagai
`forest-cover-type-dataset.zip`. Aplikasi membaca `covtype.csv` langsung dari
arsip tersebut, sehingga file tidak perlu diekstrak.

## Deploy ke Streamlit Community Cloud

1. Upload seluruh isi repository ini ke GitHub.
2. Buat aplikasi baru di Streamlit Community Cloud.
3. Pilih repository dan branch yang digunakan.
4. Isi **Main file path** dengan `mls2.py`.
5. Klik **Deploy**.

# Credit Score Classification

Proyek akhir mata kuliah Model Deployment untuk mengklasifikasikan skor kredit berdasarkan informasi keuangan dan riwayat kredit. Proyek mencakup persiapan data, perbandingan model machine learning, evaluasi, serta implementasi aplikasi menggunakan Streamlit dan layanan AWS.

**tools:** Python, Scikit-learn, XGBoost, Optuna, Streamlit, Amazon S3, Amazon SageMaker, dan Amazon EC2.

## Latar Belakang

Informasi seperti pendapatan, utang, riwayat kredit, dan perilaku pembayaran dapat digunakan untuk mempelajari pola kategori skor kredit.

Proyek ini mengeksplorasi penggunaan machine learning untuk mengklasifikasikan data ke dalam tiga kategori: **Poor, Standard, dan Good**. Hasil pengembangan model kemudian disajikan melalui aplikasi web.

## Tujuan

- Menyiapkan data keuangan untuk pemodelan.
- Membandingkan Logistic Regression, Random Forest, dan XGBoost.
- Mengevaluasi SMOTE dan pembobotan kelas untuk menangani ketidakseimbangan kelas.
- Mengoptimalkan hyperparameter menggunakan Optuna.
- Memilih model berdasarkan hasil evaluasi klasifikasi multikelas.
- Mengimplementasikan aplikasi prediksi menggunakan Streamlit dan layanan AWS.

## Dataset

Data yang digunakan dalam eksperimen berjumlah **25.000 records** dengan variabel target `Credit_Score`:

- **Poor**
- **Standard**
- **Good**

Fitur mencakup informasi pendapatan, utang, riwayat kredit, jumlah pinjaman, keterlambatan pembayaran, serta karakteristik keuangan lainnya.

Data dibagi menjadi **80% data training dan 20% data testing** menggunakan _stratified_ untuk mempertahankan proporsi kelas.

## Persiapan Data

Tahap persiapan data meliputi:

1. Menghapus kolom identifier yang tidak digunakan untuk pemodelan.
2. Menangani nilai kosong, tidak valid, dan nilai numerik yang tidak wajar.
3. Memeriksa serta menangani anomali pada data entry.
4. Melakukan transformasi pada fitur numerik tertentu.
5. Melakukan encoding variabel kategorikal.
6. Melakukan scaling fitur numerik sesuai kebutuhan model.

## Feature Engineering

Fitur tambahan dikembangkan untuk merepresentasikan hubungan antarvariabel keuangan, antara lain:

- **Debt-to-Income Ratio:** rasio utang terhadap pendapatan.
- **EMI Burden:** indikator beban cicilan.
- **Financial Exposure:** indikator eksposur keuangan.
- **Risk Score:** indikator gabungan yang dirancang dalam proyek.

Fitur tersebut merupakan hasil pengolahan untuk eksperimen dan bukan ukuran risiko kredit resmi.

## Modeling

Tiga algoritma dibandingkan:

- Logistic Regression
- Random Forest
- XGBoost

Dua pendekatan penanganan ketidakseimbangan kelas diuji, yaitu **SMOTE** dan **Class Weight**. Optimasi hyperparameter dilakukan menggunakan **Optuna** dengan validasi silang.

## Evaluasi Model

Tabel berikut menampilkan hasil eksperimen notebook pada data uji.

Precision, recall, dan F1 menggunakan **macro average**, sehingga setiap kelas memiliki bobot yang sama dalam perhitungan. ROC-AUC menggunakan pendekatan **One-vs-Rest dengan macro average**.

| Model | Pendekatan | Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | SMOTE + Optuna | 66,04% | 64,51% | 69,79% | 65,46% | 0,8111 |
| Random Forest | SMOTE + Optuna | 74,14% | 71,73% | 74,67% | 72,87% | 0,8765 |
| XGBoost | SMOTE + Optuna | 74,28% | 73,02% | 71,63% | 72,27% | 0,8757 |
| Logistic Regression | Class Weight + Optuna | 65,94% | 64,35% | 69,57% | 65,36% | 0,8132 |
| Random Forest | Class Weight + Optuna | **75,10%** | **73,52%** | 72,99% | 73,24% | **0,8812** |
| XGBoost | Class Weight + Optuna | 74,64% | 72,61% | 74,31% | **73,37%** | 0,8792 |

### Keyfindings

- **Random Forest + Class Weight + Optuna** memperoleh accuracy tertinggi sebesar **75,10%** dan ROC-AUC tertinggi sebesar **0,8812**.
- **XGBoost + Class Weight + Optuna** memperoleh macro F1 tertinggi sebesar **73,37%**.
- Di antara konfigurasi Class Weight + Optuna, XGBoost memperoleh macro recall tertinggi sebesar **74,31%**.
- Jika seluruh enam konfigurasi dibandingkan, macro recall tertinggi diperoleh **Random Forest + SMOTE + Optuna**, yaitu **74,67%**.
- Kedua model berbasis tree menunjukkan hasil lebih tinggi daripada Logistic Regression pada metrik yang ditampilkan.

## Evaluasi XGBoost per Kelas

Hasil berikut berasal dari XGBoost dengan Class Weight + Optuna.

| Kategori | Precision | Recall | F1 |
|---|---:|---:|---:|
| Poor | 0,74 | 0,77 | 0,75 |
| Standard | 0,79 | 0,75 | 0,77 |
| Good | 0,65 | 0,72 | 0,68 |

Kategori **Good** memiliki F1 paling rendah, sehingga masih menjadi bagian yang perlu ditingkatkan.

### Confusion Matrix

Baris menunjukkan kelas aktual dan kolom menunjukkan hasil prediksi.

| Aktual / Prediksi | Poor | Standard | Good |
|---|---:|---:|---:|
| Poor | 1.106 | 283 | 51 |
| Standard | 376 | 1.975 | 300 |
| Good | 15 | 243 | 651 |

Model mengklasifikasikan **3.732 dari 5.000 observasi data uji** dengan benar, sesuai accuracy sebesar **74,64%**.

## Pemilihan Final Model

Dalam eksperimen notebook, **XGBoost dengan Class Weight + Optuna** dipilih karena menghasilkan macro F1 tertinggi. Pemilihan ini mempertimbangkan performa pada seluruh kelas dalam kondisi distribusi kelas yang tidak seimbang. Random Forest tetap menjadi alternatif dengan accuracy dan ROC-AUC yang sedikit lebih tinggi.

## Implementasi Aplikasi dan AWS

Aplikasi Streamlit menyediakan formulir untuk memasukkan informasi keuangan dan menampilkan kategori skor kredit beserta probabilitas prediksi model.

Implementasi AWS yang didokumentasikan mencakup:

- **Amazon S3** untuk penyimpanan artefak model.
- **Amazon SageMaker** untuk konfigurasi layanan inferensi model.
- **Amazon EC2** untuk menjalankan aplikasi Streamlit.

Hasil evaluasi pada README ini merujuk pada eksperimen notebook. Dokumentasi deployment mencakup eksekusi terpisah, sehingga kesesuaian versi model aplikasi dengan hasil eksperimen perlu diperhatikan.

## Limitation

- Proyek dikembangkan untuk pembelajaran dan demonstrasi klasifikasi.
- Hasil evaluasi berlaku pada pembagian data dan konfigurasi eksperimen yang digunakan.
- Probabilitas keluaran model tidak otomatis menunjukkan tingkat kepastian yang telah dikalibrasi.
- Penggunaan pada data baru memerlukan evaluasi tambahan.

## Demo dan Dokumentasi

- [Buka Demo Streamlit](https://aws-demo-1-uubwmepozsrbfpza5iga7z.streamlit.app/)
- [Lihat Dokumentasi Deployment](https://drive.google.com/drive/folders/1EdFbWz2zYBAbWpGpBoyB6NfdKXqA7iUr?usp=sharing)

## Penulis

Fransciska Olivia Putri Warae  
Mahasiswa Data Science, BINUS University

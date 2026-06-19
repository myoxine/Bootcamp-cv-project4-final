# 🚦 Vehicle & Object Counter — Final Project Computer Vision

Aplikasi untuk **menghitung & mengklasifikasi** objek lalu lintas (orang, motor, mobil,
bus, truk) dari video atau **stream live** (webcam / RTSP / HLS ATCS), lengkap dengan
**arah gerak** ("dari mana ke mana"). Disertai **REST API** dan **Dockerfile**.

Menyentuh materi: Object Detection (CV 1–3), Transfer Learning (Fund. 15),
OOP & REST API (CV 11), MLOps/Docker (CV 12).

---

## Fitur
- Deteksi objek dengan **YOLOv8** (pre-trained COCO, tanpa perlu training ulang).
- **Tracking** antar-frame (ByteTrack) → tiap objek dihitung sekali.
- Dua mode:
  - **`direction_counter.py`** — klasifikasi (orang/motor/mobil/bus/truk) + **arah gerak**.
  - **`vehicle_counter.py`** — penghitung **garis** sederhana (IN/OUT) untuk satu ruas jalan.
- Sumber: file video, webcam, **RTSP**, dan **HLS `.m3u8`** (kamera ATCS Dishub).
- **REST API** (FastAPI) + **Docker**.

## Struktur folder
```
Final Project - Vehicle Counter/
├── src/                       # kode aplikasi
│   ├── direction_counter.py   # MODE UTAMA: klasifikasi + arah gerak
│   ├── vehicle_counter.py     # mode alternatif: penghitung garis IN/OUT
│   ├── api.py                 # REST API + dashboard (FastAPI) -> memakai LiveCounter
│   ├── live.py                # LiveCounter: deteksi+tracking di thread latar (dashboard)
│   ├── db.py                  # lapisan DB (PostgreSQL via SQLAlchemy)
│   └── atcs_proxy.py          # proxy penembus anti-bot ATCS (Banjarmasin/BunkerWeb)
├── web/
│   └── dashboard.html         # halaman dashboard yang disajikan di "/"
├── models/                    # bobot YOLO (yolov8n.pt / yolov8s.pt; auto-unduh bila kosong)
├── media/                     # video uji (mis. bandung_demo.mp4)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml         # app + PostgreSQL sekali jalan
└── README.md
```

---

## 1) Instalasi
```bash
pip install -r requirements.txt
```
Saat pertama jalan, bobot `yolov8n.pt` terunduh otomatis. Untuk akurasi lebih baik
(lebih lambat) ganti `--model yolov8s.pt` atau `yolov8m.pt`.

## 2) Mode utama — klasifikasi + arah (`direction_counter.py`)
```bash
# dari file video
python src/direction_counter.py --source traffic.mp4 --show

# dari stream HLS ATCS (contoh Bandung)
python src/direction_counter.py \
  --source "https://atcs-dishub.bandung.go.id:1990/SumJawa/main_stream.m3u8" \
  --show --log-every 30

# jika stream perlu header situs:
#   tambahkan --referer "https://atcs-dishub.bandung.go.id/"
```
Output akhir berupa tabel per kelas × arah, mis.:
```
motor  | Bawah -> Atas  : 14
mobil  | Kiri -> Kanan  : 6
orang  | Kanan -> Kiri  : 2
TOTAL: 22
```
Opsi penting:
- `--min-dist` : pergeseran minimum (px) agar objek dihitung (membuang yang diam/parkir).
- `--miss-frames` : berapa frame tanpa terlihat sampai track dianggap "sudah lewat".
- `--conf` : ambang confidence (turunkan ke 0.25 untuk kondisi malam/objek kecil).
- `--imgsz` : resolusi inferensi. **Di CPU, `--imgsz 320` mempercepat ±3× (≈30 FPS)** sehingga
  stream live tidak nge-lag (default 640 ≈ 11 FPS, lebih lambat dari stream 25 FPS).
- `--vid-stride` : proses tiap N frame (mis. `--vid-stride 3`). Alternatif/penambah untuk
  mengejar laju stream live di mesin lambat, atau saat memakai model lebih besar.
- `--max-frames`, `--log-every`, `--output`, `--show`.

> **Lag / video tersendat di stream live?** Akar masalahnya inferensi CPU lebih lambat dari
> laju stream (mis. 11 FPS vs 25 FPS) sehingga frame menumpuk. **Solusi utama: `--imgsz 320`** —
> mempercepat inferensi jadi ≈30 FPS (lebih cepat dari stream), sehingga **setiap frame diproses
> → video mulus DAN tidak lag**. Untuk dashboard sudah default `IMGSZ=320`, `VID_STRIDE=1`.
>
> Catatan soal `--vid-stride`: melewati frame (mis. 3) mengurangi beban/lag, **tapi membuat video
> patah-patah** karena hanya 1 dari N frame yang ditampilkan. Pakai HANYA jika `--imgsz 320` saja
> masih belum mengejar laju stream (mesin sangat lambat / pakai model besar). Untuk video mulus,
> biarkan `--vid-stride 1` dan andalkan `--imgsz`.

## 3) Mode garis sederhana (`vehicle_counter.py`)
Untuk satu ruas jalan, menghitung yang melewati satu garis (arah IN/OUT):
```bash
python src/vehicle_counter.py --source traffic.mp4 --line 0.5 --orientation horizontal --show
```

## 4) REST API (untuk file video)
```bash
uvicorn api:app --app-dir src --reload
# buka http://127.0.0.1:8000/docs lalu unggah video di POST /count
curl -X POST "http://127.0.0.1:8000/count" -F "file=@traffic.mp4"
```
> Untuk stream live (RTSP/HLS), gunakan CLI `direction_counter.py`, bukan API.

## 5) Docker
```bash
docker build -t object-counter .
docker run -p 8000:8000 object-counter
# API di http://localhost:8000/docs
```

---

## Sumber video / stream
- **File:** rekam sendiri dari jembatan penyeberangan/flyover (kamera statis) — paling
  mudah dievaluasi.
- **HLS ATCS Dishub:** cari "ATCS Dishub <kota>". Ambil URL `.m3u8` lewat
  DevTools (F12) → Network → filter `m3u8` → Copy URL. Pastikan kamu diizinkan memakainya.
- **RTSP:** `--source "rtsp://host:554/stream"` (pakai TCP lebih stabil:
  set env `OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"`).

### Dua tipe sumber ATCS (sudah diuji)
Tidak semua kamera ATCS sama. Sebelum dipakai, cek dengan `curl` apakah URL `.m3u8`
langsung membalas `#EXTM3U` (terbuka) atau dialihkan ke halaman bot-detection.

| Kamera | Anti-bot? | Cara pakai |
|---|---|---|
| **Bandung** — `https://atcs-dishub.bandung.go.id:1990/Marling/main_stream.m3u8` | Tidak (server `mediamtx`) | Arahkan **langsung** ke URL aslinya. |
| **Banjarmasin** — `https://atcs.banjarmasinkota.go.id/stream/jalan_lambung_mangkurat_bank_bi/video1_stream.m3u8` | Ya (**BunkerWeb**, proof-of-work JS) | Jalankan **`atcs_proxy.py`** dulu, lalu arahkan ke URL `localhost`. |

Contoh Bandung (tanpa proxy):
```bash
python src/direction_counter.py --source "https://atcs-dishub.bandung.go.id:1990/Marling/main_stream.m3u8" --show --conf 0.25
```

### Menembus anti-bot ATCS dengan `atcs_proxy.py`
Sebagian portal ATCS (mis. Banjarmasin) memakai anti-bot **BunkerWeb**: akses pertama
ke `.m3u8` dialihkan ke halaman *"Bot Detection"* yang menjalankan **proof-of-work
SHA-256 di JavaScript**, baru memberi cookie sesi. OpenCV/ffmpeg tidak bisa menjalankan
JS itu sehingga tak pernah menerima video. `atcs_proxy.py` mengatasinya:
1. menyelesaikan proof-of-work otomatis (cari nonce hingga `sha256(salt+nonce)` diawali `0000`),
2. menyimpan cookie sesi & **menyelesaikan ulang otomatis** saat cookie kedaluwarsa,
3. meneruskan playlist + tiap segmen ke `http://127.0.0.1:<port>/...` apa adanya.

Hanya memakai pustaka standar Python (tanpa dependency tambahan). Jalankan **2 terminal**:
```bash
# Terminal A — proxy (biarkan hidup); tunggu sampai muncul "[proxy] siap."
python src/atcs_proxy.py

# Terminal B — arahkan project ke URL localhost yang dicetak proxy
python src/direction_counter.py \
  --source "http://127.0.0.1:8899/stream/jalan_lambung_mangkurat_bank_bi/video1_stream.m3u8" \
  --show --conf 0.25
```
Kamera ATCS ber-anti-bot lain bisa dipakai dengan mengganti host/path:
```bash
python src/atcs_proxy.py --upstream "https://atcs.kotalain.go.id" \
  --path "/stream/<nama_kamera>/video1_stream.m3u8" --port 8899
```
> Catatan: penyelesaian proof-of-work awal butuh beberapa detik (mencari nonce). Untuk
> dashboard/Docker, set `VIDEO_SOURCE` ke URL `localhost` proxy, bukan URL ATCS asli.

## Evaluasi (untuk laporan & rubrik Sesi 18)
- **Akurasi hitung** vs hitungan manual pada beberapa klip uji: `1 - |pred - manual| / manual`.
- **MAE** rata-rata selisih beberapa video.
- Analisis kesalahan: occlusion, objek kecil/jauh, kondisi malam, double-count, salah arah saat membelok.
- Catat **FPS** inferensi (klaim real-time).

## Keterbatasan & tips
- **Malam hari / hujan** menurunkan akurasi; pakai model lebih besar & turunkan `--conf`.
- **Persimpangan:** mode arah memakai "arah dominan" dari lintasan — akurat untuk arus
  lurus, kurang akurat untuk yang membelok. Untuk per-lengan simpang perlu garis/zona khusus.
- Kamera harus **statis**.

## Ide pengembangan
- Mode garis/zona per lengan persimpangan (belok kiri/kanan/lurus).
- Auto-reconnect saat stream ATCS putus-nyambung.
- Estimasi kepadatan/kecepatan per menit; UI Streamlit; deploy API ke cloud.

---

## 🗄️ Database (PostgreSQL) + docker-compose

Hasil hitungan disimpan ke PostgreSQL. Tiap objek yang terhitung menjadi satu baris:
**waktu, jenis objek (orang/motor/mobil/bus/truk), arah_dari, arah_ke, track_id, sumber.**

Skema tabel:
- `sessions` — 1 baris per proses video/stream (sumber, mulai, selesai, total).
- `count_events` — 1 baris per kejadian: `waktu, kelas, arah, arah_dari, arah_ke, track_id, source`.

### Jalankan semuanya dengan docker-compose (paling mudah)
```bash
docker compose up --build
# API  : http://localhost:8000/docs
# DB   : PostgreSQL di service "db" (counter/counter, db: counterdb)
```
`docker-compose.yml` menjalankan dua service: **app** (API) + **db** (PostgreSQL),
lengkap dengan healthcheck & volume `pgdata` agar data persisten.

### Endpoint terkait DB
- `POST /count` — unggah video → hitung + simpan tiap kejadian → balikan `session_id` & ringkasan.
- `GET /events?limit=100` — daftar kejadian terbaru.
- `GET /stats` — agregasi jumlah per (kelas, arah).

### Simpan dari CLI (untuk stream live)
```bash
# set koneksi DB lalu jalankan dengan --db
export DATABASE_URL="postgresql+psycopg2://counter:counter@localhost:5432/counterdb"
python src/direction_counter.py --source "https://.../main_stream.m3u8" --db --show
```

### Contoh query SQL (untuk laporan)
```sql
-- jumlah per jenis kendaraan & arah
SELECT kelas, arah_dari, arah_ke, COUNT(*) AS jumlah
FROM count_events
GROUP BY kelas, arah_dari, arah_ke
ORDER BY jumlah DESC;

-- volume kendaraan per jam
SELECT date_trunc('hour', waktu) AS jam, COUNT(*) AS total
FROM count_events GROUP BY jam ORDER BY jam;
```

> **Tanpa Docker:** tidak perlu setup apa pun — jika `DATABASE_URL` tak di-set, otomatis
> memakai **SQLite** (`counter.db` di root). Mau pakai PostgreSQL? set `DATABASE_URL`
> (`postgresql+psycopg2://user:pass@host:5432/db`) sebelum menjalankan. Library DB:
> SQLAlchemy (+ psycopg2 untuk Postgres). Penyimpanan event live berjalan di **thread
> latar (antrian batch)** sehingga tak memblok deteksi; bila DB bermasalah, error dilog
> sekali lalu dilewati sampai pulih (video tetap mulus).

---

## 🖥️ Dashboard live (CCTV + Report + Chart)

Halaman web 3 bagian: **CCTV** (video YOLO beranotasi) | **Report** (counter semua jenis
objek + arah) di atas, dan **Chart per jam** dengan tab (per tipe objek / per arah) di bawah.

### Endpoint dashboard
- `GET /` — halaman `dashboard.html`.
- `GET /video_feed` — stream MJPEG frame beranotasi (CCTV live).
- `GET /stats` — hitungan live: `per_kelas`, `per_arah`, `total`.
- `GET /timeseries?by=tipe` — jumlah per jam per kelas (5 garis).
- `GET /timeseries?by=arah` — jumlah per jam per arah.

### Jalankan (docker-compose)
```bash
docker compose up --build
# buka http://localhost:8000/
```
Atur **sumber CCTV** lewat env `VIDEO_SOURCE` di `docker-compose.yml`:
- **YouTube**: `VIDEO_SOURCE: "https://www.youtube.com/watch?v=..."` (default contoh sudah diisi; butuh `yt-dlp`, sudah ada di `requirements.txt`)
- file: taruh di `./media/traffic.mp4` lalu set `VIDEO_SOURCE: "/media/traffic.mp4"`
- HLS ATCS: `VIDEO_SOURCE: "https://.../main_stream.m3u8"`
- RTSP: `VIDEO_SOURCE: "rtsp://host:554/stream"`
- (jika butuh header) set `REFERER`.

### Jalankan tanpa Docker
```bash
# DATABASE_URL opsional: tanpa ini otomatis pakai SQLite (counter.db). Untuk Postgres:
#   export DATABASE_URL="postgresql+psycopg2://counter:counter@localhost:5432/counterdb"
export VIDEO_SOURCE="https://atcs-dishub.bandung.go.id:1990/Buahbatu/main_stream.m3u8"  # atau "0" webcam / file / YouTube / RTSP
python -m uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000
# buka http://localhost:8000/
```

Catatan: worker live berjalan di thread latar (`live.py`), menyimpan tiap kejadian ke DB
sehingga `/stats` & `/timeseries` terisi otomatis. Chart per jam memakai `date_trunc('hour', waktu)`
(butuh PostgreSQL).

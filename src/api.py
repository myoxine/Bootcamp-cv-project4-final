"""
REST API + Dashboard untuk Vehicle & Object Counter.

Endpoint:
  GET  /             -> halaman dashboard (dashboard.html)
  GET  /video_feed   -> stream MJPEG frame YOLO beranotasi (CCTV live)
  GET  /stats        -> hitungan live: per jenis objek
  GET  /timeseries   -> jumlah per jam, per jenis objek
  POST /count        -> unggah video; hitung + simpan tiap kejadian ke DB
  GET  /events       -> daftar kejadian terbaru

Sumber CCTV diatur lewat env VIDEO_SOURCE (default "0" = webcam; bisa diisi path
file atau URL HLS/RTSP). Jalankan: uvicorn api:app  (atau docker-compose).
"""
import asyncio
import os
import shutil
import tempfile
import time
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy import func

from db import CountEvent, CountSession, SessionLocal, engine, init_db
from direction_counter import DirectionCounter
from live import LiveCounter

app = FastAPI(title="Vehicle & Object Counter Dashboard", version="3.0")
live = None
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD = os.path.join(ROOT, "web", "dashboard.html")


@app.on_event("startup")
def _startup():
    global live
    try:
        init_db()
    except Exception as e:
        print(f"[WARN] init_db gagal: {e}")
    # Tangerang CCTV (Cikupa 1, 720p) — akses langsung, tanpa proxy.
    source = os.getenv("VIDEO_SOURCE",
                       "https://cctv-dishub.tangerangkab.go.id/storage/video/01jsnyfnzs6e5818bdsed2jppp/01jsnyfnzs6e5818bdsed2jppp.m3u8?t=1781875693663")
    source = 0 if source == "0" else source
    # Untuk dashboard, batasi resolusi sumber YouTube ke 360p (lebih ringan & mulus).
    os.environ.setdefault("YT_MAX_HEIGHT", "360")
    try:
        live = LiveCounter(
            source=source,
            model=os.getenv("MODEL", "best_vehicle_sahi.pt"),  # model SAHI fine-tuned (car/motor/bus/truk)
            conf=float(os.getenv("CONF", "0.3")),
            referer=os.getenv("REFERER") or None,
            imgsz=int(os.getenv("IMGSZ", "640")),          # resolusi penuh → objek kecil/jauh terdeteksi
            vid_stride=int(os.getenv("VID_STRIDE", "1")),
            # Catatan: deteksi & tampilan terpisah (live.py) -> akurasi tinggi TANPA
            # mengorbankan kemulusan video; kotak hanya menyusul sedikit lebih lambat.
            # Ingin lebih ringan? set MODEL=yolov8n.pt, IMGSZ=320.
        )
        live.start()
    except Exception as e:
        print(f"[WARN] LiveCounter gagal start: {e}")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(DASHBOARD, encoding="utf-8") as f:
        return f.read()


@app.get("/video_feed")
def video_feed():
    """Stream MJPEG (fallback untuk <img>). Dashboard memakai /ws (canvas)."""
    def gen():
        while True:
            jpg = live.latest_jpeg if live else None
            if jpg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.05)
    return StreamingResponse(gen(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def ws_feed(ws: WebSocket):
    """Push frame JPEG ke canvas hanya saat ada frame BARU (tanpa duplikat)."""
    await ws.accept()
    last_seq = -1
    try:
        while True:
            if live and live.frame_seq != last_seq and live.latest_jpeg:
                last_seq = live.frame_seq
                await ws.send_bytes(live.latest_jpeg)
            else:
                await asyncio.sleep(0.01)   # tak ada frame baru -> cek lagi
    except WebSocketDisconnect:
        pass
    except Exception:   # noqa: BLE001  (mis. client menutup tab)
        pass


@app.get("/stats")
def stats():
    """Hitungan live: total per jenis objek."""
    if not live:
        return {"per_kelas": {}, "total": 0}
    return live.stats()   # {"per_kelas": {...}, "total": N}


@app.get("/timeseries")
def timeseries():
    """Jumlah per jam, dipecah per jenis objek (kelas)."""
    keycol = CountEvent.kelas
    # Bucket per jam: date_trunc (Postgres) vs strftime (SQLite).
    if engine.dialect.name == "postgresql":
        hour = func.date_trunc("hour", CountEvent.waktu)
    else:
        hour = func.strftime("%Y-%m-%d %H:00", CountEvent.waktu)
    with SessionLocal() as s:
        rows = (s.query(hour.label("h"), keycol.label("k"),
                        func.count().label("n"))
                .group_by(hour, keycol).order_by(hour).all())

    def hourlabel(h):  # Postgres -> datetime; SQLite -> string "...HH:00"
        return h.strftime("%H:%M") if hasattr(h, "strftime") else str(h)[11:16]

    labels_h = sorted({r.h for r in rows}, key=str)
    labels = [hourlabel(h) for h in labels_h]
    idx = {h: i for i, h in enumerate(labels_h)}
    series = {}
    for r in rows:
        series.setdefault(r.k, [0] * len(labels_h))
        series[r.k][idx[r.h]] = r.n
    return {"labels": labels, "series": series}


@app.post("/count")
async def count(file: UploadFile = File(...), conf: float = 0.3,
                min_dist: int = 40, model: str = "yolov8n.pt"):
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    events = []
    try:
        counter = DirectionCounter(model_path=model, conf=conf,
                                   min_dist=min_dist, on_count=events.append)
        result = counter.process(tmp_path, output=None, show=False)
    finally:
        os.remove(tmp_path)
    with SessionLocal() as s:
        sess = CountSession(source=file.filename)
        s.add(sess); s.commit(); s.refresh(sess)
        for ev in events:
            s.add(CountEvent(session_id=sess.id, source=file.filename, **ev))
        sess.total = result["total"]; sess.finished_at = datetime.utcnow()
        s.commit(); session_id = sess.id
    return {"session_id": session_id, "tersimpan": len(events), "hasil": result}


@app.get("/events")
def list_events(limit: int = 100):
    with SessionLocal() as s:
        rows = (s.query(CountEvent).order_by(CountEvent.id.desc())
                .limit(limit).all())
        return [{"id": r.id, "waktu": r.waktu.isoformat() if r.waktu else None,
                 "kelas": r.kelas, "track_id": r.track_id,
                 "source": r.source} for r in rows]

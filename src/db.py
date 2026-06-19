"""
Lapisan database (SQLAlchemy) untuk menyimpan hasil hitungan.

Tabel:
  sessions      -> 1 baris per proses (sumber, mulai, selesai, total)
  count_events  -> 1 baris per objek terhitung: waktu, kelas, track_id, sumber.

Koneksi diatur lewat env DATABASE_URL:
  - tanpa env (lokal/dev)  -> SQLite file counter.db (zero-setup)
  - Docker / produksi      -> postgresql+psycopg2://counter:counter@db:5432/counterdb

Penyimpanan event memakai antrian latar (enqueue_event): thread deteksi tidak
pernah ke-blok DB, penulisan di-batch, dan jika DB bermasalah error dilog SEKALI
lalu dilewati sampai pulih (tidak spam).
"""
import os
import queue
import threading
import time
from datetime import datetime

from sqlalchemy import (Column, DateTime, ForeignKey, Integer, String,
                        create_engine, func)
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import declarative_base, sessionmaker

# Tanpa DATABASE_URL (mis. dijalankan lokal tanpa Docker) -> pakai SQLite, file
# counter.db di root project; zero-setup. Docker meng-set DATABASE_URL ke Postgres.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_URL = os.getenv("DATABASE_URL",
                         "sqlite:///" + os.path.join(ROOT, "counter.db"))

# SQLite dipakai dari thread writer latar -> perlu check_same_thread=False.
_connect_args = ({"check_same_thread": False}
                 if DATABASE_URL.startswith("sqlite") else {})
engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True,
                       connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, future=True)
Base = declarative_base()


class CountSession(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    source = Column(String)                       # path/URL sumber video
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    total = Column(Integer, default=0)


class CountEvent(Base):
    __tablename__ = "count_events"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), index=True)
    waktu = Column(DateTime, default=datetime.utcnow, index=True)  # kapan dihitung
    kelas = Column(String, index=True)            # motor/mobil/bus/truk
    track_id = Column(Integer)
    source = Column(String)


def init_db(retries=15, delay=2):
    """Buat tabel; retry karena Postgres bisa belum siap saat container app start."""
    last = None
    for _ in range(retries):
        try:
            Base.metadata.create_all(engine)
            return
        except OperationalError as e:   # DB belum siap
            last = e
            time.sleep(delay)
    raise last


def create_session(source):
    with SessionLocal() as s:
        row = CountSession(source=str(source))
        s.add(row)
        s.commit()
        s.refresh(row)
        return row.id


def save_event(session_id, source, ev):
    """Tulis 1 event langsung (sinkron). Dipakai CLI; live pakai enqueue_event."""
    with SessionLocal() as s:
        s.add(CountEvent(session_id=session_id, source=str(source), **ev))
        s.commit()


# ---------------------------------------------------------------------------
# Penyimpanan latar (non-blocking, batch, anti-spam) untuk worker live.
# ---------------------------------------------------------------------------
_q = queue.Queue(maxsize=10000)   # event menunggu ditulis
_writer = None
_writer_lock = threading.Lock()
_db_ok = True                     # status koneksi terakhir (untuk log sekali)


def _flush(batch):
    """Tulis sekumpulan event dalam satu transaksi; kelola status koneksi."""
    global _db_ok
    try:
        with SessionLocal() as s:
            for session_id, source, ev in batch:
                s.add(CountEvent(session_id=session_id, source=str(source), **ev))
            s.commit()
        if not _db_ok:
            print(f"[db] koneksi pulih, penyimpanan dilanjutkan "
                  f"({len(batch)} event).", flush=True)
        _db_ok = True
    except Exception as e:   # noqa: BLE001
        if _db_ok:           # log SEKALI saat transisi ke gagal, lalu diam
            print(f"[db] gagal simpan ({type(e).__name__}): {e}. "
                  f"Event dilewati sampai DB pulih.", flush=True)
        _db_ok = False
        time.sleep(2)        # backoff; batch ini di-drop agar tak menumpuk


def _writer_loop():
    while True:
        batch = [_q.get()]            # blok sampai ada 1 event
        while len(batch) < 200:       # tarik sisanya yang sudah antre (batch)
            try:
                batch.append(_q.get_nowait())
            except queue.Empty:
                break
        _flush(batch)


def _ensure_writer():
    global _writer
    if _writer is None:
        with _writer_lock:
            if _writer is None:
                _writer = threading.Thread(target=_writer_loop, daemon=True)
                _writer.start()


def enqueue_event(session_id, source, ev):
    """Antrekan event untuk ditulis di latar (tak pernah memblok pemanggil)."""
    _ensure_writer()
    try:
        _q.put_nowait((session_id, source, ev))
    except queue.Full:
        pass   # antrian penuh (DB lambat/mati) -> jatuhkan agar live tetap mulus


def finish_session(session_id, total):
    with SessionLocal() as s:
        row = s.get(CountSession, session_id)
        if row:
            row.finished_at = datetime.utcnow()
            row.total = int(total)
            s.commit()

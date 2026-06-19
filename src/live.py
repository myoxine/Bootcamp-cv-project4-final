"""
LiveCounter - worker dashboard. VIDEO, BUFFER, dan DETEKSI dipisah agar video
tetap MULUS walau (a) inferensi YOLO lebih lambat dari stream, dan (b) stream HLS
datang dalam ledakan (burst) lalu diam.

Tiga thread:
- _capture_loop : baca stream secepat ffmpeg memberi -> masuk jitter buffer.
- _emit_loop    : keluarkan frame dari buffer pada LAJU TETAP (≈fps sumber),
                  tempel kotak deteksi terakhir, encode JPEG -> latest_jpeg (/ws).
                  Inilah yang menyerap burst HLS sehingga video tidak patah.
- _detect_loop  : jalankan YOLO pada frame yang sedang ditampilkan, laju dibatasi
                  (DETECT_FPS) agar CPU tersisa untuk tampilan.

Sumber bisa file, webcam (0), RTSP, HLS .m3u8 (ATCS), atau URL YouTube.
"""
import collections
import os
import threading
import time

import cv2

import db
from direction_counter import DirectionCounter, draw_boxes, resolve_source

# Redam log ffmpeg yang ramai (mis. "Cannot reuse HTTP connection" dari rotasi
# host googlevideo) -> warning tak berbahaya. 8 = hanya FATAL ke atas.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")

# Sisakan minimal 1 core untuk thread tampilan agar video tetap mulus saat YOLO
# memakai CPU. Tanpa ini torch memakai semua core -> tampilan kelaparan -> patah.
try:
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
except Exception:   # noqa: BLE001
    pass


class LiveCounter:
    def __init__(self, source, model="best_vehicle_seg.pt", conf=0.3,
                 min_dist=40, referer=None, imgsz=640, vid_stride=1):
        if referer:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"headers;Referer: {referer}\r\n"
        self.source = source
        self.latest_jpeg = None
        self.frame_seq = 0          # naik tiap frame baru -> WS hanya kirim yang baru
        self.session_id = None
        self._stop = False
        self._raw = None            # frame yang sedang ditampilkan (untuk deteksi)
        self._boxes = []            # kotak deteksi terakhir (dibagi ke thread emit)
        self._src_fps = 25.0        # fps sumber (diisi saat capture)
        # Jitter buffer: tampung burst HLS, dikuras emit pada laju tetap.
        self._buf = collections.deque(maxlen=int(os.getenv("BUFFER_FRAMES", "150")))
        self.counter = DirectionCounter(
            model_path=model, conf=conf, min_dist=min_dist,
            imgsz=imgsz, vid_stride=vid_stride, on_count=self._on_count)
        self._threads = [
            threading.Thread(target=self._capture_loop, daemon=True),
            threading.Thread(target=self._emit_loop, daemon=True),
            threading.Thread(target=self._detect_loop, daemon=True),
        ]

    def _on_count(self, ev):
        # Antre ke writer latar -> tak memblok; error DB dilog sekali di db.
        db.enqueue_event(self.session_id, str(self.source), ev)

    def start(self):
        try:
            self.session_id = db.create_session(str(self.source))
        except Exception as e:
            print("[live] gagal buat session:", e)
        for t in self._threads:
            t.start()

    def _capture_loop(self):
        """Baca stream secepat mungkin -> jitter buffer (auto-drop frame terlama)."""
        while not self._stop:
            url = resolve_source(self.source)
            url = 0 if url == "0" else url
            # Paksa backend FFMPEG untuk URL/stream; tanpa ini OpenCV bisa jatuh ke
            # backend CAP_IMAGES (mengira URL = pola nama file gambar) -> error aneh.
            cap = (cv2.VideoCapture(0) if url == 0
                   else cv2.VideoCapture(url, cv2.CAP_FFMPEG))
            if not cap.isOpened():
                print("[live] gagal buka sumber; coba lagi 2s")
                cap.release()
                time.sleep(2)
                continue
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            fps = cap.get(cv2.CAP_PROP_FPS)
            self._src_fps = fps if 1 < fps < 121 else 25.0
            print(f"[live] sumber terbuka, fps={self._src_fps:.0f}", flush=True)
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    break   # stream putus / file habis -> reconnect (re-resolve URL)
                self._buf.append(frame)
            cap.release()
            if not self._stop:
                time.sleep(1)

    def _emit_loop(self):
        """Keluarkan frame dari buffer pada laju tetap -> serap burst -> video mulus."""
        last = None
        t0, fcount = time.time(), 0
        while not self._stop:
            tick = time.time()
            interval = 1.0 / self._src_fps
            if self._buf:
                frame = self._buf.popleft()
                self._raw = frame          # selaras dgn yang ditampilkan -> deteksi pakai ini
                last = frame
            else:
                # buffer kosong (stall sumber) -> tahan frame terakhir, jangan spam
                time.sleep(0.01)
                continue
            disp = frame.copy()
            draw_boxes(disp, self._boxes)
            ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.latest_jpeg = buf.tobytes()
                self.frame_seq += 1
            fcount += 1
            if time.time() - t0 >= 5:
                print(f"[live] FPS tampilan: {fcount / (time.time() - t0):.1f} "
                      f"(buffer {len(self._buf)})", flush=True)
                t0, fcount = time.time(), 0
            time.sleep(max(0.0, interval - (time.time() - tick)))

    def _detect_loop(self):
        """YOLO pada frame yang sedang ditampilkan; laju dibatasi (DETECT_FPS)."""
        interval = 1.0 / max(1.0, float(os.getenv("DETECT_FPS", "6")))
        n = 0
        while not self._stop:
            frame = self._raw
            if frame is None:
                time.sleep(0.05)
                continue
            t0 = time.time()
            try:
                self._boxes = self.counter.detect(frame, n)
                n += 1
            except Exception as e:
                print("[live] error deteksi:", e)
                time.sleep(0.5)
                continue
            time.sleep(max(0.0, interval - (time.time() - t0)))

    def stop(self):
        self._stop = True

    def stats(self):
        return self.counter.summary()

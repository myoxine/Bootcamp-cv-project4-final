"""
Object Counter - deteksi + tracking lalu hitung kendaraan per JENIS
(mobil/motor/bus/truk) pada video/stream lalu lintas. Tiap objek dihitung
sekali setelah track terkonfirmasi (>= confirm_frames) agar noise tak ikut.

Callback:
  on_count(dict) -> dipanggil saat objek final dihitung {kelas, track_id}
  on_frame(bgr)  -> dipanggil tiap frame dengan gambar beranotasi (untuk streaming dashboard)
"""
import argparse
import logging
import os
from collections import defaultdict

import cv2
from ultralytics import YOLO

# Bungkam spam "WARNING Waiting for stream 0" dari loader ultralytics saat frame
# stream telat datang (gejala, bukan error). Error tetap tampil.
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Terjemahan nama kelas model (Inggris) -> label kita (Indonesia). Hanya kelas
# yang ada di sini yang dihitung. FOKUS 4 kendaraan: motor, mobil, bus, truk
# (tanpa orang). Pemetaan indeks dibangun dinamis dari model.names, jadi cocok
# untuk COCO (yolov8n/s) MAUPUN model kustom best_vehicle_seg ({0:car,1:motorcycle,2:bus,3:truck}).
TARGET_LABELS = {
    "car": "mobil", "mobil": "mobil",
    "motorcycle": "motor", "motorbike": "motor", "motor": "motor",
    "bus": "bus",
    "truck": "truk", "truk": "truk",
}

# Warna kotak per jenis objek (BGR). Disamakan dengan palet dashboard.html:
#   motor #378ADD, mobil #1D9E75, bus #7F77DD, truk #D85A30
CLASS_COLORS = {
    "motor": (221, 138, 55),
    "mobil": (117, 158, 29),
    "bus":   (221, 119, 127),
    "truk":  (48, 90, 216),
}
DEFAULT_COLOR = (0, 255, 0)

# Folder bobot YOLO; nama polos (mis. "yolov8n.pt") dicari di sini lebih dulu.
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def resolve_model(model_path):
    """Petakan nama bobot polos ke folder models/ bila ada; selain itu apa adanya
    (biar ultralytics yang mengunduh otomatis)."""
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        cand = os.path.join(MODELS_DIR, os.path.basename(model_path))
        if os.path.exists(cand):
            return cand
    return model_path


def resolve_source(source):
    """URL YouTube -> URL stream langsung (HLS) via yt-dlp.

    Default ultralytics memakai pytube yang TIDAK bisa memuat live stream
    ("is streaming live and cannot be loaded"). Kita resolve sendiri ke URL
    HLS/MP4 agar ffmpeg/OpenCV bisa membacanya. URL hasil punya masa berlaku,
    jadi fungsi ini dipanggil tiap kali process() mulai (termasuk saat reconnect).
    Sumber non-YouTube (file/RTSP/HLS/webcam) dikembalikan apa adanya.
    """
    if not (isinstance(source, str)
            and ("youtube.com" in source or "youtu.be" in source)):
        return source
    try:
        import yt_dlp
    except ImportError:
        print("[source] yt-dlp belum terpasang; pip install yt-dlp")
        return source
    try:
        # Batasi resolusi (default <=720p) agar bandwidth/decode ringan -> frame
        # live tidak telat ("Waiting for stream"). Turunkan via env mis. YT_MAX_HEIGHT=480.
        mh = os.getenv("YT_MAX_HEIGHT", "720")
        fmt = (f"best[height<={mh}][ext=mp4]/best[height<={mh}]/"
               f"best[ext=mp4]/best")
        opts = {"quiet": True, "no_warnings": True, "skip_download": True,
                "format": fmt}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(source, download=False)
        url = info.get("url") or info.get("manifest_url")
        if url:
            tag = "LIVE" if info.get("is_live") else "VIDEO"
            print(f"[source] YouTube {tag}: {info.get('title')}")
            return url
        print("[source] gagal resolve URL YouTube; pakai apa adanya.")
    except Exception as e:  # noqa: BLE001
        print(f"[source] error resolve YouTube ({e}); pakai apa adanya.")
    return source


def draw_boxes(frame, boxes):
    """Gambar kotak+label ke frame. boxes: list (x1,y1,x2,y2,label,color)."""
    for x1, y1, x2, y2, label, color in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)


class DirectionCounter:
    def __init__(self, model_path="best_vehicle_seg.pt", conf=0.3,
                 min_dist=40, miss_frames=30, on_count=None, on_frame=None,
                 imgsz=640, vid_stride=1, confirm_frames=3):
        self.model = YOLO(resolve_model(model_path))
        # Petakan indeks kelas model -> label kita, dari model.names (bukan COCO
        # tetap) agar mendukung model kustom. Hanya kelas di TARGET_LABELS dihitung.
        self.class_map = {}
        for idx, name in self.model.names.items():
            lab = TARGET_LABELS.get(str(name).lower())
            if lab:
                self.class_map[int(idx)] = lab
        self.track_classes = list(self.class_map) or None   # None = semua kelas
        self.conf = conf
        self.min_dist = min_dist      # disimpan untuk kompatibilitas (tak dipakai menghitung)
        self.miss_frames = miss_frames
        self.imgsz = imgsz            # resolusi inferensi; kecilkan (mis. 320) agar lebih cepat
        self.vid_stride = vid_stride  # proses tiap N frame; naikkan agar tak tertinggal di CPU
        # Objek dihitung sekali setelah terlihat di >= confirm_frames frame (track
        # terkonfirmasi) -> buang deteksi sekilas/noise tanpa menunda hitungan.
        self.confirm_frames = confirm_frames
        self.on_count = on_count
        self.on_frame = on_frame
        self.last_seen = {}
        self.cls_of = {}
        self.seen = defaultdict(int)     # {tid: berapa frame terlihat}
        self.counted = set()
        self.counts = defaultdict(int)   # {kelas: jumlah}

    def _maybe_count(self, tid, kelas):
        if tid in self.counted or self.seen[tid] < self.confirm_frames:
            return
        self.counts[kelas] += 1
        self.counted.add(tid)
        if self.on_count:
            self.on_count({"kelas": kelas, "track_id": int(tid)})

    def detect(self, frame, n):
        """Proses SATU frame (tracking + hitung) dan kembalikan daftar kotak untuk
        digambar: (x1, y1, x2, y2, label, color).

        Dipakai pipeline live yang memisahkan deteksi dari tampilan: thread deteksi
        memanggil ini sesanggupnya CPU; thread tampilan menempelkan kotaknya ke tiap
        frame. persist=True menjaga ID tracker meski sebagian frame dilewati.
        """
        results = self.model.track(
            frame, persist=True, classes=self.track_classes, conf=self.conf,
            imgsz=self.imgsz, tracker="bytetrack.yaml", verbose=False)
        out = []
        present = set()
        b = results[0].boxes
        if b is not None and b.id is not None:
            ids = b.id.int().cpu().tolist()
            clss = b.cls.int().cpu().tolist()
            xyxy = b.xyxy.cpu().numpy()
            for tid, cls, (x1, y1, x2, y2) in zip(ids, clss, xyxy):
                present.add(tid)
                self.last_seen[tid] = n
                self.seen[tid] += 1
                kelas = self.class_map.get(cls, str(cls))
                self.cls_of[tid] = kelas
                self._maybe_count(tid, kelas)
                color = CLASS_COLORS.get(kelas, DEFAULT_COLOR)
                out.append((int(x1), int(y1), int(x2), int(y2),
                            f"{kelas} #{tid}", color))
        for tid, seen in list(self.last_seen.items()):
            if tid not in present and (n - seen) > self.miss_frames:
                self.last_seen.pop(tid, None)
                self.cls_of.pop(tid, None)
                self.seen.pop(tid, None)
        return out

    def process(self, source, output=None, show=False, fps=25,
                max_frames=0, log_every=0):
        source = resolve_source(source)
        writer = None
        w = h = None
        n = 0
        try:
            for result in self.model.track(
                source=source, stream=True, persist=True,
                classes=self.track_classes, conf=self.conf,
                imgsz=self.imgsz, vid_stride=self.vid_stride,
                tracker="bytetrack.yaml", verbose=False,
            ):
                frame = result.orig_img
                if frame is None:
                    continue
                if w is None:
                    h, w = frame.shape[:2]
                    if output:
                        writer = cv2.VideoWriter(
                            output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

                present = set()
                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    ids = boxes.id.int().cpu().tolist()
                    clss = boxes.cls.int().cpu().tolist()
                    xyxy = boxes.xyxy.cpu().numpy()
                    for tid, cls, (x1, y1, x2, y2) in zip(ids, clss, xyxy):
                        present.add(tid)
                        self.last_seen[tid] = n
                        self.seen[tid] += 1
                        kelas = self.class_map.get(cls, str(cls))
                        self.cls_of[tid] = kelas
                        self._maybe_count(tid, kelas)   # hitung saat terkonfirmasi
                        color = CLASS_COLORS.get(kelas, DEFAULT_COLOR)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                      color, 2)
                        cv2.putText(frame, f"{kelas} #{tid}",
                                    (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, color, 2)

                # bersihkan track yang sudah lama hilang (counted tetap disimpan
                # agar tak terhitung ulang bila ID-nya muncul lagi)
                for tid, seen in list(self.last_seen.items()):
                    if tid not in present and (n - seen) > self.miss_frames:
                        self.last_seen.pop(tid, None)
                        self.cls_of.pop(tid, None)
                        self.seen.pop(tid, None)

                if self.on_frame:
                    self.on_frame(frame)
                if writer:
                    writer.write(frame)
                if show:
                    cv2.imshow("Direction Counter", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                n += 1
                if log_every and n % log_every == 0:
                    print(f"[frame {n}] total terhitung={len(self.counted)}")
                if max_frames and n >= max_frames:
                    break
        except KeyboardInterrupt:
            print("\nDihentikan oleh pengguna (Ctrl+C).")

        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        return self.summary()

    def summary(self):
        out = dict(self.counts)
        return {"per_kelas": out, "total": sum(out.values())}


def main():
    ap = argparse.ArgumentParser(description="Hitung objek per jenis pada video/stream")
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--output", default=None)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="resolusi inferensi; kecilkan (mis. 320) untuk lebih cepat di CPU")
    ap.add_argument("--vid-stride", type=int, default=1,
                    help="proses tiap N frame (mis. 3) agar tak tertinggal di stream live")
    ap.add_argument("--min-dist", type=int, default=40)
    ap.add_argument("--miss-frames", type=int, default=30)
    ap.add_argument("--referer", default=None)
    ap.add_argument("--db", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=30)
    args = ap.parse_args()

    if args.referer:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"headers;Referer: {args.referer}\r\n"

    on_count = None
    dbmod = None
    db_session_id = None
    if args.db:
        import db as dbmod
        dbmod.init_db()
        db_session_id = dbmod.create_session(args.source)

        def on_count(ev):
            dbmod.save_event(db_session_id, args.source, ev)

    source = 0 if args.source == "0" else args.source
    counter = DirectionCounter(args.model, args.conf, args.min_dist,
                               args.miss_frames, on_count=on_count,
                               imgsz=args.imgsz, vid_stride=args.vid_stride)
    result = counter.process(source, output=args.output, show=args.show,
                             max_frames=args.max_frames, log_every=args.log_every)
    if args.db and dbmod is not None:
        dbmod.finish_session(db_session_id, result["total"])
    print("\n=== Hasil: jumlah per jenis ===")
    for kelas, jml in result["per_kelas"].items():
        print(f"  {kelas:6s} : {jml}")
    print(f"TOTAL: {result['total']}")


if __name__ == "__main__":
    main()

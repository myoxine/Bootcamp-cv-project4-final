"""
Vehicle Counter - menghitung kendaraan yang melewati sebuah garis virtual pada video.

Pipeline: YOLOv8 (deteksi) -> tracking (ByteTrack, ID antar-frame) -> hitung saat
titik tengah objek melintasi garis. Gaya OOP (class VehicleCounter), materi CV 11.

Contoh:
    python vehicle_counter.py --source traffic.mp4 --output hasil.mp4
    python vehicle_counter.py --source 0                       # webcam
    python vehicle_counter.py --source "rtsp://host/stream"    # stream RTSP
"""
import argparse
from collections import defaultdict

import cv2
from ultralytics import YOLO

from direction_counter import resolve_model, resolve_source

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class VehicleCounter:
    """Mendeteksi, melacak, dan menghitung kendaraan yang melintasi satu garis."""

    def __init__(self, model_path="best_vehicle_seg.pt", conf=0.3,
                 line_position=0.5, orientation="horizontal",
                 imgsz=640, vid_stride=1):
        self.model = YOLO(resolve_model(model_path))
        self.conf = conf
        self.line_position = line_position
        self.orientation = orientation
        self.imgsz = imgsz            # resolusi inferensi; kecilkan (mis. 320) agar lebih cepat
        self.vid_stride = vid_stride  # proses tiap N frame; naikkan agar tak tertinggal di CPU
        self.last_pos = {}
        self.counted = set()
        self.count_in = defaultdict(int)
        self.count_out = defaultdict(int)

    def _line_value(self, w, h):
        if self.orientation == "horizontal":
            return int(h * self.line_position)
        return int(w * self.line_position)

    def _axis_value(self, cx, cy):
        return cy if self.orientation == "horizontal" else cx

    def process(self, source, output=None, show=False, fps=25,
                max_frames=0, log_every=0):
        """Proses video/stream (termasuk RTSP); kembalikan ringkasan hitungan.

        max_frames > 0 : berhenti setelah sejumlah frame (untuk stream live).
        log_every  > 0 : cetak hitungan berjalan tiap sekian frame.
        Ctrl+C untuk berhenti; ringkasan tetap dikembalikan.
        """
        source = resolve_source(source)
        writer = None
        line = None
        w = h = None
        n = 0
        try:
            for result in self.model.track(
                source=source, stream=True, persist=True,
                classes=list(VEHICLE_CLASSES), conf=self.conf,
                imgsz=self.imgsz, vid_stride=self.vid_stride,
                tracker="bytetrack.yaml", verbose=False,
            ):
                frame = result.orig_img
                if frame is None:
                    continue
                if w is None:
                    h, w = frame.shape[:2]
                    line = self._line_value(w, h)
                    if output:
                        writer = cv2.VideoWriter(
                            output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    ids = boxes.id.int().cpu().tolist()
                    clss = boxes.cls.int().cpu().tolist()
                    xyxy = boxes.xyxy.cpu().numpy()
                    for tid, cls, (x1, y1, x2, y2) in zip(ids, clss, xyxy):
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        cur = self._axis_value(cx, cy)
                        prev = self.last_pos.get(tid)
                        self.last_pos[tid] = cur
                        if prev is not None and tid not in self.counted:
                            name = VEHICLE_CLASSES.get(cls, str(cls))
                            if prev < line <= cur:
                                self.count_in[name] += 1
                                self.counted.add(tid)
                            elif prev > line >= cur:
                                self.count_out[name] += 1
                                self.counted.add(tid)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                      (0, 255, 0), 2)
                        cv2.putText(frame, f"{VEHICLE_CLASSES.get(cls, cls)} #{tid}",
                                    (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 1)

                if line is not None:
                    if self.orientation == "horizontal":
                        cv2.line(frame, (0, line), (w, line), (0, 0, 255), 2)
                    else:
                        cv2.line(frame, (line, 0), (line, h), (0, 0, 255), 2)
                total_in = sum(self.count_in.values())
                total_out = sum(self.count_out.values())
                cv2.putText(frame, f"IN: {total_in}  OUT: {total_out}",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                if writer:
                    writer.write(frame)
                if show:
                    cv2.imshow("Vehicle Counter", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                n += 1
                if log_every and n % log_every == 0:
                    print(f"[frame {n}] IN={total_in} OUT={total_out} "
                          f"TOTAL={total_in + total_out}")
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
        total_in = sum(self.count_in.values())
        total_out = sum(self.count_out.values())
        return {
            "per_kelas_in": dict(self.count_in),
            "per_kelas_out": dict(self.count_out),
            "total_in": total_in,
            "total_out": total_out,
            "total": total_in + total_out,
        }


def main():
    ap = argparse.ArgumentParser(
        description="Hitung kendaraan yang melewati garis pada video/stream")
    ap.add_argument("--source", required=True,
                    help="path video, '0' untuk webcam, atau URL RTSP/HTTP stream")
    ap.add_argument("--model", default="best_vehicle_seg.pt", help="bobot YOLOv8")
    ap.add_argument("--output", default=None, help="video hasil anotasi (opsional)")
    ap.add_argument("--conf", type=float, default=0.3, help="ambang confidence")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="resolusi inferensi; kecilkan (mis. 320) untuk lebih cepat di CPU")
    ap.add_argument("--vid-stride", type=int, default=1,
                    help="proses tiap N frame (mis. 3) agar tak tertinggal di stream live")
    ap.add_argument("--line", type=float, default=0.5, help="posisi garis (0..1)")
    ap.add_argument("--orientation", choices=["horizontal", "vertical"],
                    default="horizontal")
    ap.add_argument("--show", action="store_true", help="tampilkan preview")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="berhenti setelah N frame (0 = tak terbatas; untuk stream)")
    ap.add_argument("--log-every", type=int, default=30,
                    help="cetak hitungan berjalan tiap N frame (0 = nonaktif)")
    args = ap.parse_args()

    source = 0 if args.source == "0" else args.source
    counter = VehicleCounter(args.model, args.conf, args.line, args.orientation,
                             imgsz=args.imgsz, vid_stride=args.vid_stride)
    result = counter.process(source, output=args.output, show=args.show,
                             max_frames=args.max_frames, log_every=args.log_every)
    print("\n=== Hasil Hitung Kendaraan ===")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

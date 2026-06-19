"""
Vet sumber live: ambil 1 frame, laporkan RESOLUSI + KECERAHAN (siang/malam),
dan simpan preview. Berguna untuk menilai kandidat stream sebelum dipakai.

Pakai:
    python tools/check_source.py "https://.../main_stream.m3u8"
    python tools/check_source.py "https://www.youtube.com/watch?v=..."   # YouTube juga bisa
    python tools/check_source.py 0                                        # webcam
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
from direction_counter import resolve_source


def main():
    if len(sys.argv) < 2:
        print('pakai: python tools/check_source.py "<url / youtube / 0>"')
        return
    raw = sys.argv[1]
    url = resolve_source(raw)
    url = 0 if url == "0" else url

    cap = (cv2.VideoCapture(0) if url == 0
           else cv2.VideoCapture(url, cv2.CAP_FFMPEG))
    if not cap.isOpened():
        print("GAGAL buka sumber (mati / token kedaluwarsa / butuh header).")
        return

    frame = None
    for _ in range(15):           # buang frame awal, ambil yang stabil
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()

    if frame is None:
        print("Terbuka, tapi tak ada frame (stream kosong/stall).")
        return

    h, w = frame.shape[:2]
    bright = float(frame.mean())
    siang = "SIANG (terang)" if bright > 70 else "gelap (sore/malam)"
    cv2.imwrite("source_preview.jpg", frame)
    print(f"resolusi  : {w}x{h}")
    print(f"kecerahan : {bright:.0f}/255  ->  {siang}")
    print("preview   : source_preview.jpg (buka untuk lihat kualitas & sudut)")


if __name__ == "__main__":
    main()

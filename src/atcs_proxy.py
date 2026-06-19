"""
ATCS HLS proxy - menembus anti-bot BunkerWeb pada CCTV ATCS lalu meneruskan
stream HLS (.m3u8 + segmen .mp4) ke localhost agar bisa dibaca OpenCV/ffmpeg.

Kenapa perlu ini?
  CCTV ATCS Banjarmasin dilindungi anti-bot "BunkerWeb": akses pertama ke URL
  .m3u8 dialihkan ke halaman "Bot Detection" yang menjalankan proof-of-work
  SHA-256 di JavaScript, baru memberi cookie sesi. OpenCV/ffmpeg tak bisa
  menjalankan JS itu, jadi tak pernah dapat videonya. Proxy ini:
    1) menyelesaikan proof-of-work otomatis (cari nonce hingga sha256 diawali "0000"),
    2) menyimpan cookie sesi & menyelesaikan ULANG saat cookie kedaluwarsa,
    3) meneruskan playlist + tiap segmen ke http://127.0.0.1:<port>/... apa adanya.

Pakai:
    python atcs_proxy.py            # default: kamera Lambung Mangkurat (Bank BI)
    # lalu di terminal lain arahkan project ke URL localhost yang dicetak, mis:
    #   python direction_counter.py --source "http://127.0.0.1:8899/stream/jalan_lambung_mangkurat_bank_bi/video1_stream.m3u8" --show

Ganti kamera lewat --upstream "<host>" --path "/stream/.../video1_stream.m3u8".
"""
import argparse
import hashlib
import http.cookiejar
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urljoin, urlsplit

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# Diisi dari argumen CLI saat start.
UPSTREAM = ""          # mis. "https://atcs.banjarmasinkota.go.id"
REFERER = ""           # mis. "https://atcs.banjarmasinkota.go.id/"
TRIGGER_PATH = "/"     # path stream yang dipakai untuk memicu halaman challenge

_cj = http.cookiejar.CookieJar()
_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_cj))
_solve_lock = threading.Lock()

_POW_RE = re.compile(r'digestMessage\("([^"]+)"\+a\.toString\(\)\)\)\.startsWith\("([0-9a-f]+)"\)')
_FIELD_RE = re.compile(r'<input[^>]+name="([^"]+)"[^>]+id="challenge"')
_ACTION_RE = re.compile(r'<form[^>]+action="([^"]+)"')


def _headers():
    return {"User-Agent": UA, "Referer": REFERER, "Accept": "*/*"}


def _is_challenge(url, body_head):
    return url.endswith("/challenge") or b"Bot Detection" in body_head


def solve_challenge():
    """Selesaikan proof-of-work sekali; cookie tersimpan di _cj otomatis."""
    with _solve_lock:
        # Akses path stream akan dialihkan (302) ke /challenge; urllib mengikuti
        # redirect otomatis sehingga kita menerima halaman proof-of-work.
        req = urllib.request.Request(UPSTREAM + TRIGGER_PATH, headers=_headers())
        try:
            resp = _opener.open(req, timeout=25)
        except urllib.error.HTTPError as e:
            resp = e
        page_url = resp.geturl()
        html = resp.read().decode("utf-8", "replace")
        m = _POW_RE.search(html)
        if not m:
            print("[proxy] tidak menemukan pola proof-of-work; cookie mungkin sudah valid.")
            return
        salt, prefix = m.group(1), m.group(2)
        field = (_FIELD_RE.search(html) or [None, "challenge"])[1]
        action = (_ACTION_RE.search(html) or [None, "/challenge"])[1]

        nonce = 0
        while not hashlib.sha256(f"{salt}{nonce}".encode()).hexdigest().startswith(prefix):
            nonce += 1

        data = urllib.parse.urlencode({field: str(nonce)}).encode()
        post = urllib.request.Request(
            urljoin(page_url, action), data=data, headers={
                **_headers(), "Content-Type": "application/x-www-form-urlencoded"})
        _opener.open(post, timeout=25).read()
        print(f"[proxy] anti-bot terlewati (nonce={nonce}). Cookie: "
              f"{[c.name for c in _cj]}", flush=True)


def fetch_upstream(path):
    """Ambil `path` dari upstream dengan cookie; selesaikan ulang challenge bila perlu.

    Return (status, content_type, body_bytes).
    """
    url = UPSTREAM + path
    for attempt in range(2):
        req = urllib.request.Request(url, headers=_headers())
        try:
            resp = _opener.open(req, timeout=25)
            body = resp.read()
            if _is_challenge(resp.geturl(), body[:200]):
                solve_challenge()
                continue
            return resp.status, resp.headers.get("Content-Type", "application/octet-stream"), body
        except urllib.error.HTTPError as e:
            body = e.read()
            if e.code in (302, 403) or _is_challenge(getattr(e, "url", ""), body[:200]):
                solve_challenge()
                continue
            return e.code, e.headers.get("Content-Type", "text/plain"), body
        except Exception as e:  # noqa: BLE001
            return 502, "text/plain", f"proxy error: {e}".encode()
    return 502, "text/plain", b"proxy: gagal setelah retry"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):  # noqa: N802
        path = urlsplit(self.path).path
        if path in ("/", "/favicon.ico"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"ATCS proxy aktif. Arahkan player ke path .m3u8 kamera.")
            return
        status, ctype, body = fetch_upstream(path)
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # kurangi spam log
        if "video1_stream.m3u8" in (self.path or ""):
            print("[proxy]", self.address_string(), fmt % args)


def main():
    global UPSTREAM, REFERER, TRIGGER_PATH
    ap = argparse.ArgumentParser(description="Proxy HLS penembus anti-bot ATCS")
    ap.add_argument("--upstream", default="https://atcs.banjarmasinkota.go.id",
                    help="skema+host CCTV ATCS")
    ap.add_argument("--path",
                    default="/stream/jalan_lambung_mangkurat_bank_bi/video1_stream.m3u8",
                    help="path playlist .m3u8 (hanya untuk dicetak sebagai contoh)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    args = ap.parse_args()

    UPSTREAM = args.upstream.rstrip("/")
    REFERER = UPSTREAM + "/"
    TRIGGER_PATH = args.path

    print(f"[proxy] upstream : {UPSTREAM}")
    print("[proxy] menyelesaikan anti-bot awal ...")
    try:
        solve_challenge()
    except Exception as e:  # noqa: BLE001
        print(f"[proxy] peringatan: challenge awal gagal ({e}); akan dicoba saat request.")

    local = f"http://{args.host}:{args.port}{args.path}"
    print(f"[proxy] siap. Arahkan project ke:\n    {local}")
    print(f"[proxy] contoh:\n    python direction_counter.py --source \"{local}\" --show")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

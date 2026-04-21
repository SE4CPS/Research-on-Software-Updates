#!/usr/bin/env python3

import http.client
import http.server
import os
import socketserver
import sys
import urllib.parse
from pathlib import Path

script_dir = Path(__file__).parent
os.chdir(script_dir)
PORT = 8000

UPSTREAM = "releasetrain.io"


class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/releasetrain-api/"):
            self._proxy_releasetrain(parsed)
            return
        super().do_GET()

    def _proxy_releasetrain(self, parsed):
        rest = self.path[len("/releasetrain-api/") :].lstrip("/")
        if not rest:
            self.send_error(404, "Missing path after /releasetrain-api/")
            return
        # Mirror Netlify: /releasetrain-api/<path> -> https://releasetrain.io/api/<path>
        target_path = "/api/" + rest
        if parsed.query:
            target_path += "?" + parsed.query
        try:
            conn = http.client.HTTPSConnection(UPSTREAM, timeout=120)
            conn.request("GET", target_path, headers={"User-Agent": "research-open-text-sentimental-local/1.0"})
            resp = conn.getresponse()
            body = resp.read()
            self.send_response(resp.status)
            ct = resp.getheader("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError as e:
            self.send_error(502, f"Proxy error: {e}")


if __name__ == "__main__":
    try:
        with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
            print(f"Server running at: http://localhost:{PORT}")
            print("Proxy: /releasetrain-api/* -> https://releasetrain.io/api/*")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

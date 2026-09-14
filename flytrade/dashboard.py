"""Read-only, loopback-only dashboard; no dependencies or external CDN."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
from threading import Thread

from .config import Config
from .research import load_report

LOG = logging.getLogger(__name__)


def snapshot(config: Config) -> dict:
    paper_path = config.data_dir / "paper.json"
    paper = json.loads(paper_path.read_text()) if paper_path.exists() else None
    try:
        report = load_report(config, paper["run_id"] if paper else None)
    except (FileNotFoundError, ValueError):
        report = None
    progress = config.data_dir / "progress.json"
    return {"paper": paper, "report": report,
            "progress": json.loads(progress.read_text()) if progress.exists() else None}


def start_dashboard(config: Config):
    page = (Path(__file__).parent / "web" / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                body, content_type = page, "text/html; charset=utf-8"
            elif self.path == "/api/status":
                try:
                    body = json.dumps(snapshot(config), allow_nan=False).encode()
                except (OSError, ValueError):
                    self.send_error(503, "Status temporarily unavailable")
                    return
                content_type = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", config.app.dashboard_port), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    LOG.info("Dashboard: http://127.0.0.1:%d", config.app.dashboard_port)
    return server

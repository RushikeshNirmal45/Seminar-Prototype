import mimetypes
import os

BASE_DIR = os.path.dirname(__file__)
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
HTML_FILE = os.path.join(PUBLIC_DIR, "index.html")


def app(environ, start_response):
    """WSGI entrypoint for Vercel Python runtime serving the Cyber Guardian AI application."""
    path = environ.get("PATH_INFO", "/").lstrip("/")

    # Static file serving (e.g. assets/hero_illustration.png)
    if path.startswith("assets/"):
        file_path = os.path.join(PUBLIC_DIR, path)
        if os.path.isfile(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            status = "200 OK"
            headers = [("Content-Type", mime_type or "application/octet-stream")]
            start_response(status, headers)
            with open(file_path, "rb") as f:
                return [f.read()]

    # Main application HTML page
    status = "200 OK"
    response_headers = [("Content-Type", "text/html; charset=utf-8")]
    start_response(status, response_headers)
    if os.path.exists(HTML_FILE):
        with open(HTML_FILE, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = "<!DOCTYPE html><html><body><h1>Cyber Guardian AI</h1></body></html>"
    return [content.encode("utf-8")]


application = app


import os

HTML_FILE = os.path.join(os.path.dirname(__file__), "public", "index.html")


def app(environ, start_response):
    """WSGI entrypoint for Vercel Python runtime."""
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

#!/usr/bin/env python3
"""Bearer-key reverse proxy for an inference server that has no API-key option of its own (veloGB10, ...).

Usage:
  python3 auth-proxy.py --listen 0.0.0.0:8000 --upstream http://127.0.0.1:8001 --key-file ~/spark-llm/.api_key

- Every request needs `Authorization: Bearer <key>` (the whole stripped content of --key-file, as speedtest.py
  reads it), except `GET /health`. A missing or wrong key gets HTTP 401 with a JSON error body.
- The key check compares SHA-256 digests with hmac.compare_digest: constant time, and it does not leak the length.
- Responses are relayed as they arrive: one read1() of the upstream body, one write to the client, with TCP_NODELAY on
  both sockets. A body without Content-Length (SSE, chunked) is re-chunked to the client piece by piece, so
  time-to-first-token and per-token timing measured through the proxy match the upstream's.
- Request bodies are read whole (Content-Length or chunked) up to --max-body (default 64 MiB), then forwarded with an
  exact Content-Length; a larger body gets 413.
- One thread per connection (ThreadingHTTPServer), HTTP/1.1 keep-alive towards the client, a fresh upstream
  connection per request. The Authorization header is not forwarded upstream. Python 3.12 standard library only.
"""
import argparse
import hashlib
import hmac
import http.client
import http.server
import json
import os
import signal
import socket
import sys
import time
from urllib.parse import urlsplit

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "trailers",
              "transfer-encoding", "upgrade", "expect", "proxy-connection"}
PIECE = 64 * 1024           # largest single relay read
UNAUTH_DRAIN = 1 << 20      # body bytes read (and dropped) before a 401, so the client sees the reply, not a reset


def load_key(path):
    """Return the key file's stripped content as bytes; exit if it is empty or holds whitespace."""
    with open(os.path.expanduser(path), "rb") as f:
        key = f.read().strip()
    if not key or any(c in key for c in b" \t\r\n"):
        sys.exit(f"auth-proxy: {path} must hold exactly one non-empty key")
    return key


def parse_listen(text):
    """'HOST:PORT' -> (host, port); IPv6 hosts may be bracketed."""
    host, _, port = text.rpartition(":")
    return host.strip("[]") or "0.0.0.0", int(port)


class Proxy(http.server.BaseHTTPRequestHandler):
    """One client connection; the server object carries the configuration (key digest, upstream, limits)."""

    protocol_version = "HTTP/1.1"
    server_version = "auth-proxy"
    sys_version = ""
    timeout = 900               # an idle client connection is closed after 15 minutes

    # ---- logging -------------------------------------------------------------------------------------------------
    def log_message(self, fmt, *args):
        """Access lines on stderr with a timestamp; never the key."""
        sys.stderr.write(f"{time.strftime('%F %T')} {self.client_address[0]} {fmt % args}\n")

    # ---- auth ----------------------------------------------------------------------------------------------------
    def authorized(self):
        """True when the Authorization header carries the configured bearer key (constant-time comparison)."""
        scheme, _, token = (self.headers.get("Authorization") or "").strip().partition(" ")
        if scheme.lower() != "bearer":
            return False
        # http.client/http.server decode header values as ISO-8859-1, so latin-1 recovers the bytes that were sent.
        got = hashlib.sha256(token.strip().encode("latin-1", "replace")).digest()
        return hmac.compare_digest(got, self.server.key_digest)

    def is_open_route(self):
        """GET /health (any query string) is the one route that needs no key."""
        return self.command == "GET" and self.path.split("?", 1)[0] == "/health"

    def handle_expect_100(self):
        """Refuse an unauthenticated or oversized `Expect: 100-continue` upload before the client sends its body."""
        if not (self.is_open_route() or self.authorized()):
            self.reply_json(401, "invalid_api_key", "missing or invalid API key", close=True,
                            extra_headers=[("WWW-Authenticate", "Bearer")])
            return False
        try:
            too_big = int(self.headers.get("Content-Length") or 0) > self.server.max_body
        except ValueError:
            too_big = False
        if too_big:
            self.reply_json(413, "body_too_large", f"request body over {self.server.max_body} bytes", close=True)
            return False
        return super().handle_expect_100()

    # ---- small replies -------------------------------------------------------------------------------------------
    def reply_json(self, status, code, message, close=False, extra_headers=()):
        """Send a complete JSON error reply with an exact Content-Length."""
        body = json.dumps({"error": {"message": message, "type": "invalid_request_error" if status < 500
                                     else "server_error", "code": code}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra_headers:
            self.send_header(k, v)
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    # ---- request body --------------------------------------------------------------------------------------------
    def read_body(self, limit):
        """The request body as bytes (Content-Length or chunked), or None after replying 413/400 to the client."""
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            parts, total = [], 0
            while True:
                line = self.rfile.readline(1024)
                try:
                    size = int(line.split(b";", 1)[0].strip(), 16)
                except ValueError:
                    self.reply_json(400, "bad_chunk", "malformed chunked request body", close=True)
                    return None
                if size == 0:
                    while self.rfile.readline(1024) not in (b"\r\n", b"\n", b""):
                        pass                                  # trailers, ignored
                    return b"".join(parts)
                total += size
                if total > limit:
                    self.reply_json(413, "body_too_large", f"request body over {limit} bytes", close=True)
                    return None
                parts.append(self.rfile.read(size))
                self.rfile.readline(1024)                     # the CRLF after each chunk
        length = self.headers.get("Content-Length")
        if length is None:
            return b""
        try:
            n = int(length)
        except ValueError:
            self.reply_json(400, "bad_length", "malformed Content-Length", close=True)
            return None
        if n < 0 or n > limit:
            self.reply_json(413, "body_too_large", f"request body over {limit} bytes", close=True)
            return None
        body = self.rfile.read(n)
        if len(body) != n:
            self.close_connection = True
            return None
        return body

    def drain_unauthorized(self):
        """Drop up to UNAUTH_DRAIN bytes of an unauthenticated body; True when it was read completely."""
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            return False
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return False
        if n > UNAUTH_DRAIN:
            return False
        self.rfile.read(n)
        return True

    # ---- the proxy -----------------------------------------------------------------------------------------------
    def relay(self):
        """Check the key, forward the request upstream, stream the reply back."""
        if not self.is_open_route() and not self.authorized():
            complete = self.drain_unauthorized()
            self.reply_json(401, "invalid_api_key", "missing or invalid API key", close=not complete,
                            extra_headers=[("WWW-Authenticate", "Bearer")])
            return
        body = self.read_body(self.server.max_body)
        if body is None:
            return
        up = self.server.upstream
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in HOP_BY_HOP and k.lower() not in ("host", "content-length", "authorization")}
        headers["Host"] = up.netloc
        if body or self.command in ("POST", "PUT", "PATCH"):
            headers["Content-Length"] = str(len(body))
        path = (up.path.rstrip("/") + self.path) if up.path not in ("", "/") else self.path
        conn = http.client.HTTPConnection(up.hostname, up.port or 80, timeout=self.server.upstream_timeout)
        try:
            try:
                conn.connect()
                conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.request(self.command, path, body=body if body or "Content-Length" in headers else None,
                             headers=headers)
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException) as e:
                self.reply_json(502, "upstream_unavailable", f"upstream {up.netloc}: {e.__class__.__name__}: {e}")
                return
            self.log_request(resp.status)
            self.send_response_only(resp.status, resp.reason)
            self.send_header("Date", self.date_time_string())
            drop = HOP_BY_HOP | {h.strip().lower() for h in (resp.getheader("Connection") or "").split(",")}
            for k, v in resp.getheaders():
                if k.lower() not in drop and k.lower() not in ("content-length", "date", "server"):
                    self.send_header(k, v)
            no_body = self.command == "HEAD" or resp.status in (204, 304) or 100 <= resp.status < 200
            length = resp.getheader("Content-Length")
            chunked = not no_body and length is None
            if chunked and self.request_version != "HTTP/1.1":
                chunked, self.close_connection = False, True   # an HTTP/1.0 client reads the body to EOF
            if no_body:
                if length is not None:
                    self.send_header("Content-Length", length)
            elif chunked:
                self.send_header("Transfer-Encoding", "chunked")
            elif length is not None:
                self.send_header("Content-Length", length)
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            if no_body:
                return
            w = self.wfile
            try:
                while True:
                    piece = resp.read1(PIECE)
                    if not piece:
                        break
                    if chunked:
                        w.write(b"%x\r\n%s\r\n" % (len(piece), piece))
                    else:
                        w.write(piece)
                    w.flush()
                if chunked:
                    w.write(b"0\r\n\r\n")
                    w.flush()
            except (http.client.HTTPException, OSError) as e:
                # Upstream cut mid-body or the client left: the framing is broken, so drop the client connection.
                self.log_message("relay of %s %s ended early: %s: %s", self.command, self.path,
                                 e.__class__.__name__, e)
                self.close_connection = True
        finally:
            conn.close()

    def setup(self):
        super().setup()
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = relay


class Server(http.server.ThreadingHTTPServer):
    """Threaded server holding the proxy's configuration."""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def handle_error(self, request, client_address):
        """A client that resets or times out between requests is routine: one line, not a traceback."""
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, TimeoutError)):
            sys.stderr.write(f"{time.strftime('%F %T')} {client_address[0]} connection closed: "
                             f"{exc.__class__.__name__}\n")
            return
        super().handle_error(request, client_address)


def stop(*_):
    """SIGTERM handler: unwind serve_forever() in the main thread like Ctrl-C."""
    raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--listen", required=True, help="HOST:PORT to accept clients on, e.g. 0.0.0.0:8000")
    ap.add_argument("--upstream", required=True, help="base URL of the keyless server, e.g. http://127.0.0.1:8001")
    ap.add_argument("--key-file", required=True, help="file holding the bearer key (its whole stripped content)")
    ap.add_argument("--max-body", type=int, default=64 << 20, help="largest request body in bytes (default 64 MiB)")
    ap.add_argument("--upstream-timeout", type=float, default=7200.0,
                    help="seconds one upstream socket operation may wait (a 200K-token prefill can be minutes)")
    a = ap.parse_args()
    up = urlsplit(a.upstream)
    if up.scheme != "http" or not up.hostname:
        sys.exit("auth-proxy: --upstream must be an http:// URL")
    srv = Server(parse_listen(a.listen), Proxy)
    srv.key_digest = hashlib.sha256(load_key(a.key_file)).digest()
    srv.upstream, srv.max_body, srv.upstream_timeout = up, a.max_body, a.upstream_timeout
    signal.signal(signal.SIGTERM, stop)
    sys.stderr.write(f"{time.strftime('%F %T')} auth-proxy listening on {a.listen} -> {a.upstream} "
                     f"(bearer key required except GET /health)\n")
    sys.stderr.flush()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()

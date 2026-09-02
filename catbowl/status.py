"""A tiny status page, so you can check the rig from your phone.

Deliberately stdlib-only: http://<pi>:8080/ shows each bowl's state, and
/snapshot/<bowl>.jpg shows exactly what that camera is looking at, which is how
you find out the lens has been nudged sideways.

It also accepts POST /control to pin a lid open or closed by hand. There is no
authentication of any kind, so anyone who can reach the port can work the lids.
That is a deliberate trade for a box on a home LAN; do not port-forward it, and
set status_port: null if the network is shared.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger(__name__)

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>catbowl</title>
<style>
 body{font:15px/1.5 system-ui,sans-serif;margin:0;padding:1.2rem;background:#14161a;color:#e8e6e3}
 h1{font-size:1.1rem;margin:0 0 1rem;color:#9aa0a6;font-weight:600;letter-spacing:.02em}
 .bowl{background:#1d2026;border:1px solid #2c313a;border-radius:10px;padding:1rem;margin-bottom:1rem}
 .row{display:flex;justify-content:space-between;gap:1rem;align-items:baseline}
 .cat{font-size:1.15rem;font-weight:600}
 .state{font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;padding:.15rem .5rem;border-radius:99px}
 .open{background:#1e4620;color:#8fd694}.closed{background:#2c313a;color:#9aa0a6}.cooldown{background:#4a3a1a;color:#e0c07a}
 dl{display:grid;grid-template-columns:auto 1fr;gap:.2rem .8rem;margin:.8rem 0 0;font-size:.85rem;color:#9aa0a6}
 dd{margin:0;color:#e8e6e3}
 img{width:100%;max-width:320px;border-radius:6px;margin-top:.8rem;background:#000}
 table{width:100%;border-collapse:collapse;font-size:.82rem}
 td{padding:.25rem .5rem .25rem 0;border-bottom:1px solid #2c313a;color:#9aa0a6}
 .err{color:#e08c8c}
 .btns{display:flex;gap:.5rem;margin-top:.8rem}
 button{flex:1;font:inherit;font-size:.85rem;padding:.45rem .5rem;border-radius:6px;
   border:1px solid #3a4150;background:#252a33;color:#e8e6e3;cursor:pointer}
 button:hover{background:#2f3540}
 button.on{background:#1e4620;border-color:#2f6b34;color:#8fd694}
 .held{font-size:.78rem;color:#e0c07a;margin-top:.5rem}
</style>
<h1>catbowl <span id=up></span></h1><div id=bowls></div><h1>recent</h1><table id=events></table>
<script>
async function tick(){
 const s = await (await fetch('/status.json')).json();
 up.textContent = '· up ' + Math.floor(s.uptime_s/60) + 'm' + (s.no_model ? ' · NO MODEL: opens for any cat' : '');
 bowls.innerHTML = s.bowls.map(b => `
  <div class=bowl>
   <div class=row><span class=cat>${b.cat}</span><span class="state ${b.state}">${b.state}</span></div>
   <dl><dt>bowl<dd>${b.bowl}<dt>lid<dd>${Math.round(b.lid*100)}%<dt>seeing<dd>${b.seen} (${b.confidence})
   <dt>opens today<dd>${b.opens} · ${Math.round(b.seconds_open)}s<dt>denials<dd>${b.denials}
   ${b.error ? '<dt>error<dd class=err>'+b.error : ''}</dl>
   <div class=btns>
    <button class="${b.manual==='open'?'on':''}" onclick="hold('${b.bowl}','open')">open</button>
    <button class="${b.manual==='closed'?'on':''}" onclick="hold('${b.bowl}','closed')">close</button>
    <button onclick="hold('${b.bowl}',null)" ${b.manual?'':'disabled'}>auto</button>
   </div>
   ${b.manual ? '<div class=held>held '+b.manual+' by hand - press auto to resume</div>' : ''}
   <img src="/snapshot/${b.bowl}.jpg?t=${Date.now()}" alt="">
  </div>`).join('');
 events.innerHTML = s.recent_events.map(e =>
   `<tr><td>${e.time}<td>${e.bowl}<td>${e.kind}<td>${e.cat||''}<td>${JSON.stringify(e.detail)}</tr>`).join('');
}
async function hold(bowl, lid){
 await fetch('/control', {method:'POST', headers:{'Content-Type':'application/json'},
                          body: JSON.stringify({bowl, lid})});
 tick();
}
tick(); setInterval(tick, 2000);
</script>
"""


def _handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/", "/index.html"):
                    self._send(PAGE.encode(), "text/html; charset=utf-8")
                elif path == "/status.json":
                    self._send(json.dumps(app.status(), default=str).encode(), "application/json")
                elif path.startswith("/snapshot/") and path.endswith(".jpg"):
                    self._snapshot(path[len("/snapshot/"):-len(".jpg")])
                else:
                    self._send(b"not found", "text/plain", 404)
            except BrokenPipeError:  # pragma: no cover - browser navigated away
                pass
            except Exception:
                log.exception("status request failed: %s", self.path)
                self._send(b"error", "text/plain", 500)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path.split("?", 1)[0] != "/control":
                self._send(b"not found", "text/plain", 404)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                bowl = body.get("bowl")
                lid = body.get("lid")          # "open" | "closed" | null for auto
                app.set_manual(bowl, lid)
            except KeyError:
                self._send(b"no such bowl", "text/plain", 404)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send(str(exc).encode(), "text/plain", 400)
            except BrokenPipeError:  # pragma: no cover - browser navigated away
                pass
            except Exception:
                log.exception("control request failed")
                self._send(b"error", "text/plain", 500)
            else:
                self._send(json.dumps(app.status(), default=str).encode(),
                           "application/json")

        def _snapshot(self, bowl_id: str) -> None:
            import cv2

            worker = next((w for w in app.workers if w.cfg.id == bowl_id), None)
            frame = worker.latest_frame if worker else None
            if frame is None:
                self._send(b"no frame", "text/plain", 404)
                return
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ok:
                self._send(b"encode failed", "text/plain", 500)
                return
            self._send(buf.tobytes(), "image/jpeg")

        def log_message(self, *args) -> None:   # keep the console for real logs
            pass

    return Handler


def _lan_address() -> str:
    """The address this machine is actually reachable at from the network.

    Logging 0.0.0.0 is useless: you cannot type it into a phone. Connecting a
    UDP socket to an off-machine address sends nothing, but it makes the kernel
    pick the interface it would route through, and its local address is the one
    worth printing.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 9))     # TEST-NET-1: reserved, never routed
        return sock.getsockname()[0]
    except OSError:
        return socket.gethostname()        # no network; the hostname may resolve
    finally:
        sock.close()


def start_status_server(app, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", port), _handler_for(app))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="status", daemon=True).start()

    import socket

    urls = [f"http://{_lan_address()}:{port}/"]
    host = socket.gethostname()
    if host and not host.startswith("localhost"):
        urls.append(f"http://{host}.local:{port}/")
    log.info("status page on %s", "  or  ".join(urls))
    return server

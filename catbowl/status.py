"""A tiny status page, so you can check the rig from your phone.

Deliberately stdlib-only: http://<pi>:8080/ shows each bowl's state, and
/snapshot/<bowl>.jpg shows exactly what that camera is looking at, which is how
you find out the lens has been nudged sideways.

/sort is the other half: the photos the rig banks, one at a time, with a button
per label, so a week of captures can be filed from a phone on the sofa. It is
kept deliberately cheap - the page never polls, filenames arrive in batches, an
image is served straight off the disk without being decoded, and a label is a
rename. The feeder's own threads should not notice it is there.

It also accepts POST /control to pin a lid open or closed by hand. There is no
authentication of any kind, so anyone who can reach the port can work the lids.
That is a deliberate trade for a box on a home LAN; do not port-forward it, and
set status_port: null if the network is shared.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from .sorting import PAGE_SIZE, SortError

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
<h1>catbowl <span id=up></span></h1><div id=bowls></div>
<p><a href="/sort">sort captured photos</a> · <a href="/browse">browse</a></p>
<h1>recent</h1><table id=events></table>
<script>
async function tick(){
 const s = await (await fetch('/status.json')).json();
 up.textContent = '· up ' + Math.floor(s.uptime_s/60) + 'm' + (s.no_model ? ' · NO MODEL: opens for any cat' : '');
 bowls.innerHTML = s.bowls.map(b => `
  <div class=bowl>
   <div class=row><span class=cat>${b.cat}</span><span class="state ${b.state}">${b.state}</span></div>
   <dl><dt>bowl<dd>${b.bowl}<dt>lid<dd>${Math.round(b.lid*100)}%<dt>seeing<dd>${b.seen} (${b.confidence})
   <dt>opens today<dd>${b.opens} · ${Math.round(b.seconds_open)}s<dt>denials<dd>${b.denials}
   ${b.captured ? '<dt>photos kept<dd>'+b.captured : ''}
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


SORT_PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>catbowl · sort</title>
<style>
 body{font:15px/1.5 system-ui,sans-serif;margin:0;padding:1rem;background:#14161a;color:#e8e6e3;
   -webkit-user-select:none;user-select:none}
 h1{font-size:1.1rem;margin:0 0 .8rem;color:#9aa0a6;font-weight:600}
 h1 a{color:#9aa0a6}
 #shot{width:100%;max-width:420px;aspect-ratio:1;object-fit:contain;background:#000;
   border-radius:10px;display:block;margin:0 auto}
 #name{font-size:.75rem;color:#6b7280;text-align:center;margin:.4rem 0 .8rem;
   font-family:ui-monospace,monospace}
 .keys{display:flex;flex-wrap:wrap;gap:.5rem;max-width:420px;margin:0 auto}
 .keys button{flex:1 1 4rem;font:inherit;font-size:1.2rem;font-weight:600;padding:.9rem .5rem;
   border-radius:8px;border:1px solid #3a4150;background:#252a33;color:#e8e6e3;cursor:pointer}
 .keys button:active{background:#1e4620;border-color:#2f6b34}
 .keys button.minor{font-size:.9rem;font-weight:400;color:#9aa0a6}
 #tally{max-width:420px;margin:1rem auto 0;font-size:.82rem;color:#9aa0a6;
   display:flex;flex-wrap:wrap;gap:.8rem;justify-content:center}
 #done{text-align:center;color:#9aa0a6;padding:3rem 1rem}
</style>
<h1><a href="/">&larr; catbowl</a> · sort · <a href="/browse">browse</a></h1>
<div id=app></div>
<div id=tally></div>
<script>
let queue = [], labels = [], counts = {}, busy = false, filling = false;

async function fill(force){
 if (filling) return;                     // one listing in flight at a time
 filling = true;
 try {
  const response = await fetch('/sort/queue.json' + (force ? '?refresh=1' : ''));
  if (!response.ok){                      // capture is off, or the folder vanished
   app.textContent = await response.text();
   return;
  }
  const r = await response.json();
  labels = r.labels; counts = r.counts;
  // Keep anything already on screen; add only names we have not seen.
  const seen = new Set(queue);
  for (const n of r.pending) if (!seen.has(n)) queue.push(n);
  draw();
 } finally { filling = false; }
}

// Top up after a decision - never from draw(). draw() calling fill() calling
// draw() spins forever as soon as the folder holds fewer photos than one batch,
// which is precisely the end of every sorting session.
function topUp(){ if (queue.length < 4) fill(true); }

function draw(){
 const totals = labels.concat(['discard']).map(l => l + ' ' + (counts[l]||0));
 tally.textContent = totals.join('  ·  ') + '   left: ' + (counts.unsorted||0);
 if (!queue.length){
  app.innerHTML = '<div id=done>Nothing left to sort.<br><br>' +
    '<button class=minor onclick="fill(true)">check again</button></div>';
  return;
 }
 const buttons = labels.map(l =>
   `<button onclick="pick('${l}')">${l}</button>`).join('') +
   `<button class=minor onclick="pick('discard')">junk</button>` +
   `<button class=minor onclick="skip()">skip</button>` +
   `<button class=minor onclick="undo()">undo</button>`;
 app.innerHTML = `<img id=shot src="/sort/photo/${queue[0]}" alt="">
   <div id=name>${queue[0]}</div><div class=keys>${buttons}</div>`;
 // Fetch the next two now, while a human is deciding about this one. They are
 // served with a long cache lifetime, so showing them costs no request.
 for (const n of queue.slice(1, 3)) new Image().src = '/sort/photo/' + n;
}

async function pick(label){
 if (busy || !queue.length) return;
 busy = true;
 const name = queue.shift();
 try {
  const r = await fetch('/sort/label', {method:'POST', headers:{'Content-Type':'application/json'},
                                        body: JSON.stringify({name, label})});
  const body = await r.json();
  if (body.counts) counts = body.counts;
 } catch (e) { queue.unshift(name); }
 busy = false;
 draw();
 topUp();
}

function skip(){ if (queue.length){ queue.push(queue.shift()); draw(); } }

async function undo(){
 if (busy) return;
 busy = true;
 const r = await (await fetch('/sort/undo', {method:'POST'})).json();
 if (r.name) queue.unshift(r.name);
 if (r.counts) counts = r.counts;
 busy = false;
 draw();
 topUp();
}

addEventListener('keydown', e => {
 if (e.key === 'u') return undo();
 if (e.key === 'x') return pick('discard');
 if (e.key === ' ') { e.preventDefault(); return skip(); }
 const hit = labels.find(l => l.toLowerCase() === e.key.toLowerCase());
 if (hit) pick(hit);
});

fill(true);
</script>
"""


BROWSE_PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>catbowl · browse</title>
<style>
 body{font:15px/1.5 system-ui,sans-serif;margin:0;padding:1rem;background:#14161a;color:#e8e6e3}
 h1{font-size:1.1rem;margin:0 0 .8rem;color:#9aa0a6;font-weight:600}
 h1 a{color:#9aa0a6}
 .tabs{display:flex;flex-wrap:wrap;gap:.4rem;margin-bottom:.8rem}
 .tabs button{font:inherit;font-size:.9rem;padding:.4rem .7rem;border-radius:99px;
   border:1px solid #3a4150;background:#252a33;color:#9aa0a6;cursor:pointer}
 .tabs button.on{background:#1e4620;border-color:#2f6b34;color:#8fd694}
 #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:.4rem}
 #grid img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;background:#000;
   cursor:pointer;border:2px solid transparent}
 #grid img.on{border-color:#8fd694}
 .pager{display:flex;gap:.5rem;justify-content:center;align-items:center;margin:1rem 0;
   color:#9aa0a6;font-size:.85rem}
 .pager button{font:inherit;padding:.4rem .8rem;border-radius:6px;border:1px solid #3a4150;
   background:#252a33;color:#e8e6e3;cursor:pointer}
 .pager button[disabled]{opacity:.4;cursor:default}
 #sheet{position:fixed;left:0;right:0;bottom:0;background:#1d2026;border-top:1px solid #2c313a;
   padding:.8rem;display:none}
 #sheet.show{display:block}
 #sheet .who{font-size:.75rem;color:#6b7280;font-family:ui-monospace,monospace;
   text-align:center;margin-bottom:.6rem;overflow-wrap:anywhere}
 #sheet .keys{display:flex;flex-wrap:wrap;gap:.5rem;max-width:520px;margin:0 auto}
 #sheet button{flex:1 1 4rem;font:inherit;font-weight:600;padding:.7rem .4rem;border-radius:8px;
   border:1px solid #3a4150;background:#252a33;color:#e8e6e3;cursor:pointer}
 #sheet button.minor{font-weight:400;color:#9aa0a6}
 #empty{color:#9aa0a6;text-align:center;padding:2rem}
</style>
<h1><a href="/sort">&larr; sort</a> · browse</h1>
<div class=tabs id=tabs></div>
<div id=grid></div>
<div class=pager id=pager></div>
<div id=sheet><div class=who id=who></div><div class=keys id=keys></div></div>
<script>
let buckets = [], counts = {}, bucket = 'unsorted', offset = 0, limit = 40,
    total = 0, page = [], picked = null, busy = false;

async function load(){
 const r = await (await fetch(`/sort/browse.json?bucket=${bucket}&offset=${offset}`)).json();
 buckets = r.buckets; page = r.names; total = r.total; offset = r.offset;
 limit = r.limit; picked = null;
 draw();
}

function draw(){
 tabs.innerHTML = buckets.map(b =>
   `<button class="${b===bucket?'on':''}" onclick="go('${b}')">${b} ${counts[b]??''}</button>`).join('');
 // Newest first, so a photo just filed by mistake is in the top-left corner.
 grid.innerHTML = page.length
  ? page.map(n => `<img loading=lazy src="/sort/photo/${n}?bucket=${bucket}"
      class="${n===picked?'on':''}" onclick="tap('${n}')" alt="${n}">`).join('')
  : '<div id=empty>This folder is empty.</div>';
 const last = Math.max(0, total - 1);
 pager.innerHTML = total > page.length || offset
  ? `<button onclick="hop(-1)" ${offset ? '' : 'disabled'}>newer</button>
     <span>${total ? offset+1 : 0}-${offset+page.length} of ${total}</span>
     <button onclick="hop(1)" ${offset+page.length > last ? 'disabled' : ''}>older</button>`
  : `<span>${total} photo${total===1?'':'s'}</span>`;
 sheet.className = picked ? 'show' : '';
 who.textContent = picked || '';
 keys.innerHTML = buckets.filter(b => b !== bucket).map(b =>
   `<button onclick="move('${b}')">${b}</button>`).join('') +
   `<button class=minor onclick="tap(null)">close</button>`;
}

function go(b){ if (b !== bucket){ bucket = b; offset = 0; load(); } }
function hop(dir){ offset = Math.max(0, offset + dir * limit); load(); }
function tap(n){ picked = (n === picked) ? null : n; draw(); }

async function move(target){
 if (busy || !picked) return;
 busy = true;
 const name = picked;
 try {
  const r = await fetch('/sort/move', {method:'POST', headers:{'Content-Type':'application/json'},
                                       body: JSON.stringify({name, from: bucket, to: target})});
  if (r.ok){
   const body = await r.json();
   counts = body.counts || counts;
   page = page.filter(x => x !== name);   // no reload: one rename, one less tile
   total -= 1;
   picked = null;
  }
 } finally { busy = false; }
 draw();
}

(async () => {
 counts = await (await fetch('/sort/queue.json')).json().then(r => r.counts).catch(() => ({}));
 load();
})();
</script>
"""


def _handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, body: bytes, content_type: str, status: int = 200,
                  cache: str = "no-store") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, status: int = 200) -> None:
            self._send(json.dumps(payload, default=str).encode(), "application/json", status)

        def _sorter(self):
            """The sorter, or None once a 404 has been sent explaining why."""
            if app.sorter is None:
                self._send(b"photo capture is off (capture.dir is not set)", "text/plain", 404)
                return None
            return app.sorter

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/", "/index.html"):
                    self._send(PAGE.encode(), "text/html; charset=utf-8")
                elif path == "/status.json":
                    self._send(json.dumps(app.status(), default=str).encode(), "application/json")
                elif path.startswith("/snapshot/") and path.endswith(".jpg"):
                    self._snapshot(path[len("/snapshot/"):-len(".jpg")])
                elif path == "/sort":
                    self._send(SORT_PAGE.encode(), "text/html; charset=utf-8")
                elif path == "/browse":
                    self._send(BROWSE_PAGE.encode(), "text/html; charset=utf-8")
                elif path == "/sort/browse.json":
                    self._browse()
                elif path == "/sort/queue.json":
                    self._queue("refresh=1" in (self.path.split("?", 1) + [""])[1])
                elif path.startswith("/sort/photo/"):
                    self._photo(path[len("/sort/photo/"):])
                else:
                    self._send(b"not found", "text/plain", 404)
            except (BrokenPipeError, ConnectionResetError):  # pragma: no cover - client left
                pass
            except Exception:
                log.exception("status request failed: %s", self.path)
                self._send(b"error", "text/plain", 500)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = self.path.split("?", 1)[0]
            if path == "/sort/label":
                self._label()
                return
            if path == "/sort/undo":
                self._undo()
                return
            if path == "/sort/move":
                self._move()
                return
            if path != "/control":
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
            except (BrokenPipeError, ConnectionResetError):  # pragma: no cover - client left
                pass
            except Exception:
                log.exception("control request failed")
                self._send(b"error", "text/plain", 500)
            else:
                self._send(json.dumps(app.status(), default=str).encode(),
                           "application/json")

        def _queue(self, refresh: bool) -> None:
            sorter = self._sorter()
            if sorter is None:
                return
            self._json({"labels": sorter.labels,
                        "pending": sorter.pending(refresh=refresh),
                        "counts": sorter.counts()})

        def _browse(self) -> None:
            sorter = self._sorter()
            if sorter is None:
                return
            query = parse_qs(urlsplit(self.path).query)
            bucket = (query.get("bucket") or ["unsorted"])[0]
            try:
                offset = max(0, int((query.get("offset") or ["0"])[0]))
                names, total = sorter.listing(bucket, offset)
            except SortError as exc:
                self._send(str(exc).encode(), "text/plain", 400)
                return
            except ValueError:
                self._send(b"offset must be a number", "text/plain", 400)
                return
            self._json({"bucket": bucket, "buckets": sorter.all_buckets, "names": names,
                        "total": total, "offset": offset, "limit": PAGE_SIZE})

        def _move(self) -> None:
            sorter = self._sorter()
            if sorter is None:
                return
            try:
                body = self._read_json()
                landed = sorter.move(str(body.get("name", "")),
                                     str(body.get("from", "")), str(body.get("to", "")))
            except (SortError, ValueError, TypeError) as exc:
                self._send(str(exc).encode(), "text/plain", 400)
            except (BrokenPipeError, ConnectionResetError):  # pragma: no cover - client left
                pass
            except Exception:
                log.exception("move request failed")
                self._send(b"error", "text/plain", 500)
            else:
                self._json({"name": landed, "counts": sorter.counts()})

        def _photo(self, name: str) -> None:
            """Serve one captured photo straight off the disk.

            No decode, no re-encode: these are the bytes cv2 wrote. The long
            cache lifetime is safe because a capture filename is a timestamp and
            is never reused, and it is what makes the page's preloading free.
            """
            sorter = self._sorter()
            if sorter is None:
                return
            bucket = (parse_qs(urlsplit(self.path).query).get("bucket") or ["unsorted"])[0]
            try:
                path = sorter.path_for(name, bucket)
            except SortError as exc:
                self._send(str(exc).encode(), "text/plain", 400)
                return
            try:
                body = path.read_bytes()
            except OSError:
                self._send(b"no such photo", "text/plain", 404)
                return
            self._send(body, "image/jpeg", cache="public, max-age=31536000, immutable")

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("expected a JSON object")
            return body

        def _label(self) -> None:
            sorter = self._sorter()
            if sorter is None:
                return
            try:
                body = self._read_json()
                filed = sorter.assign(str(body.get("name", "")), str(body.get("label", "")))
            except (SortError, ValueError, TypeError) as exc:
                self._send(str(exc).encode(), "text/plain", 400)
            except (BrokenPipeError, ConnectionResetError):  # pragma: no cover - client left
                pass
            except Exception:
                log.exception("sort request failed")
                self._send(b"error", "text/plain", 500)
            else:
                self._json({"filed": filed, "counts": sorter.counts()})

        def _undo(self) -> None:
            sorter = self._sorter()
            if sorter is None:
                return
            self._json({"name": sorter.undo(), "counts": sorter.counts()})

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


class _QuietServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that does not shout when a browser hangs up.

    The page polls every two seconds over keep-alive connections, so a phone
    locking its screen, a tab closing, or a reload mid-request leaves a socket
    the client has already reset. socketserver's default handler prints the
    whole traceback to stderr for each one, which buries the feeder's own log in
    ConnectionResetError. Anything else still gets reported in full.
    """

    def handle_error(self, request, client_address) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, TimeoutError)):
            log.debug("status client %s hung up: %s", client_address[0], type(exc).__name__)
            return
        log.exception("status request from %s failed", client_address[0])


def start_status_server(app, port: int) -> ThreadingHTTPServer:
    server = _QuietServer(("0.0.0.0", port), _handler_for(app))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="status", daemon=True).start()

    import socket

    urls = [f"http://{_lan_address()}:{port}/"]
    host = socket.gethostname()
    if host and not host.startswith("localhost"):
        urls.append(f"http://{host}.local:{port}/")
    log.info("status page on %s", "  or  ".join(urls))
    return server

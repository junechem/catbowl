"""Filing captured photos, and the /sort endpoints that drive it from a phone."""

import json
import urllib.error
import urllib.request

import pytest

from catbowl.sorting import DISCARD, SortError, Sorter

LABELS = ["J", "K", "F", "M"]


@pytest.fixture
def collected(tmp_path):
    unsorted = tmp_path / "unsorted"
    unsorted.mkdir()
    for i in range(5):
        (unsorted / f"bowl1-2026090{i}-120000-000.jpg").write_bytes(b"\xff\xd8jpeg" + bytes([i]))
    return tmp_path


@pytest.fixture
def sorter(collected):
    return Sorter(collected, LABELS)


def test_pending_lists_the_captures_oldest_first(sorter):
    names = sorter.pending()
    assert len(names) == 5
    assert names == sorted(names)


def test_labelling_moves_the_file(sorter, collected):
    name = sorter.pending()[0]
    sorter.assign(name, "K")

    assert (collected / "K" / name).is_file()
    assert not (collected / "unsorted" / name).exists()
    assert name not in sorter.pending(), "a filed photo must leave the queue"
    assert sorter.counts()["K"] == 1
    assert sorter.counts()["unsorted"] == 4


def test_discarding_keeps_the_file(sorter, collected):
    """'Junk' is a folder, not a delete - the user was explicit about that."""
    name = sorter.pending()[0]
    sorter.assign(name, DISCARD)
    assert (collected / DISCARD / name).is_file()


def test_undo_puts_the_last_photo_back(sorter, collected):
    name = sorter.pending()[0]
    sorter.assign(name, "J")
    assert sorter.undo() == name

    assert (collected / "unsorted" / name).is_file()
    assert not (collected / "J" / name).exists()
    assert sorter.pending()[0] == name
    assert sorter.undo() is None, "undo is one step deep, and says so"


def test_a_name_collision_does_not_overwrite_hand_labelled_work(sorter, collected):
    name = sorter.pending()[0]
    sorter.assign(name, "F")
    (collected / "unsorted" / name).write_bytes(b"\xff\xd8second")

    filed = sorter.assign(name, "F")
    assert filed != name
    assert (collected / "F" / name).read_bytes().startswith(b"\xff\xd8jpeg")
    assert (collected / "F" / filed).read_bytes() == b"\xff\xd8second"


@pytest.mark.parametrize("name", ["../../etc/passwd", "..%2fx.jpg", "/etc/shadow.jpg",
                                  "sub/dir.jpg", "no-extension", ""])
def test_a_path_outside_the_folder_is_refused(sorter, name):
    with pytest.raises(SortError):
        sorter.path_for(name)


def test_an_unknown_label_is_refused(sorter):
    with pytest.raises(SortError):
        sorter.assign(sorter.pending()[0], "Z")


def test_a_missing_photo_is_refused(sorter):
    with pytest.raises(SortError):
        sorter.assign("bowl9-20260101-000000-000.jpg", "J")


def test_an_empty_folder_is_not_an_error(tmp_path):
    empty = Sorter(tmp_path / "nothing", LABELS)
    assert empty.pending() == []
    assert empty.counts()["unsorted"] == 0


# --------------------------------------------------------------------------- #
# the HTTP surface
# --------------------------------------------------------------------------- #

class FakeApp:
    """Just enough FeederApp for the status server."""

    workers: list = []

    def __init__(self, sorter):
        self.sorter = sorter

    def status(self):
        return {"version": "test", "uptime_s": 1, "no_model": True,
                "cameras": {}, "bowls": [], "recent_events": []}

    def set_manual(self, bowl_id, mode):
        raise KeyError(bowl_id)


@pytest.fixture
def server(sorter):
    from catbowl.status import start_status_server

    srv = start_status_server(FakeApp(sorter), 8097)
    yield "http://127.0.0.1:8097"
    srv.shutdown()


def get(url):
    with urllib.request.urlopen(url) as response:
        return response.status, response.read(), dict(response.headers)


def post(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else b""
    request = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return response.status, json.loads(response.read())


def test_the_sort_page_is_served(server):
    """The page is a static shell; the buttons come from the config via JSON."""
    status, body, _ = get(f"{server}/sort")
    assert status == 200
    assert b"/sort/queue.json" in body and b"/sort/label" in body


def test_the_queue_lists_labels_and_pending_photos(server):
    status, body, _ = get(f"{server}/sort/queue.json")
    payload = json.loads(body)
    assert status == 200
    assert payload["labels"] == LABELS
    assert len(payload["pending"]) == 5
    assert payload["counts"]["unsorted"] == 5


def test_a_photo_is_served_verbatim_and_cacheable(server, collected):
    name = sorted(p.name for p in (collected / "unsorted").glob("*.jpg"))[0]
    status, body, headers = get(f"{server}/sort/photo/{name}")
    assert status == 200
    assert body == (collected / "unsorted" / name).read_bytes(), "bytes must not be re-encoded"
    assert "immutable" in headers["Cache-Control"]


def test_labelling_over_http_moves_the_file(server, collected):
    name = sorted(p.name for p in (collected / "unsorted").glob("*.jpg"))[0]
    status, payload = post(f"{server}/sort/label", {"name": name, "label": "M"})
    assert status == 200
    assert payload["counts"]["M"] == 1
    assert (collected / "M" / name).is_file()

    _, payload = post(f"{server}/sort/undo")
    assert payload["name"] == name
    assert (collected / "unsorted" / name).is_file()


def test_a_traversal_attempt_over_http_is_a_400(server):
    with pytest.raises(urllib.error.HTTPError) as caught:
        get(f"{server}/sort/photo/..%2f..%2fetc%2fpasswd")
    assert caught.value.code == 400


def test_an_unknown_label_over_http_is_a_400(server, collected):
    name = sorted(p.name for p in (collected / "unsorted").glob("*.jpg"))[0]
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(f"{server}/sort/label", {"name": name, "label": "Z"})
    assert caught.value.code == 400
    assert (collected / "unsorted" / name).is_file(), "a rejected label must move nothing"


def test_sorting_is_404_when_capture_is_off():
    from catbowl.status import start_status_server

    srv = start_status_server(FakeApp(None), 8096)
    try:
        with pytest.raises(urllib.error.HTTPError) as caught:
            get("http://127.0.0.1:8096/sort/queue.json")
        assert caught.value.code == 404
    finally:
        srv.shutdown()

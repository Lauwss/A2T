"""Round-trip tests for the local listening-test server."""

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from a2t.data.wav import write_wav_24

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def load_server_module():
    spec = importlib.util.spec_from_file_location(
        "listening_server", SCRIPTS_DIR / "listening_server.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def server(tmp_path):
    module = load_server_module()
    data_dir = tmp_path / "stimuli"
    data_dir.mkdir()
    results_dir = tmp_path / "results"
    html = tmp_path / "page.html"
    html.write_text("<html>ok</html>", encoding="utf-8")
    write_wav_24(data_dir / "masker_only.wav", 0.1 * np.sin(2 * np.pi * 1000.0 * np.arange(4800) / 48000), 48000)
    (data_dir / "manifest.json").write_text(json.dumps({"version": 1}), encoding="utf-8")
    (data_dir / "answers.json").write_text(json.dumps({"levels": {}}), encoding="utf-8")

    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0), module.make_handler(html, data_dir, results_dir)
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, results_dir
    httpd.shutdown()
    httpd.server_close()


def get(base: str, path: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(base + path, timeout=5) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_serves_page_and_stimuli(server) -> None:
    base, _ = server
    code, body = get(base, "/")
    assert code == 200
    assert b"ok" in body

    code, body = get(base, "/manifest.json")
    assert code == 200
    assert json.loads(body)["version"] == 1

    code, body = get(base, "/masker_only.wav")
    assert code == 200
    assert len(body) > 100


def test_missing_and_outside_files_return_404(server) -> None:
    base, _ = server
    assert get(base, "/nope.wav")[0] == 404
    assert get(base, "/sub/manifest.json")[0] == 404
    assert get(base, "/answers_extra.json")[0] == 404
    assert get(base, "/some.html")[0] == 404


def test_post_results_saves_json(server) -> None:
    base, results_dir = server
    payload = json.dumps({"version": 1, "trials": []}).encode()
    req = urllib.request.Request(
        base + "/results", data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=5) as res:
        body = json.loads(res.read())
    saved = list(results_dir.glob("*.json"))
    assert body["saved"] == saved[0].name
    assert json.loads(saved[0].read_text(encoding="utf-8"))["version"] == 1


def test_post_results_rejects_bad_body(server) -> None:
    base, results_dir = server
    req = urllib.request.Request(
        base + "/results", data=b"not json", headers={"Content-Type": "application/json"}, method="POST"
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req, timeout=5)
    assert excinfo.value.code == 400
    assert not list(results_dir.glob("*.json"))

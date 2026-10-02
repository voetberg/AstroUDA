import functools
import hashlib
import json
import logging
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator, Optional

import pytest

from astrouda.cli import build_argument_parser, main
from astrouda.data import cli_registration as cli_registration_module
from astrouda.data import download as download_module
from astrouda.data.download import (
    DATASET_SOURCES,
    ChecksumMismatchError,
    DownloadError,
    InsufficientDiskSpaceError,
    RemoteFile,
    UnknownFileError,
    default_data_directory,
    download_dataset,
    download_file,
    fetch_record_files,
)

RECORD_ID: str = "123"
FILE_CONTENTS: dict[str, bytes] = {
    "alpha.bin": bytes(range(256)) * 40,
    "beta.bin": b"beta-content" * 500,
    "gamma.bin": b"beta-content" * 500,
}


class FakeZenodoServer:
    def __init__(self, directory: Path) -> None:
        self.directory: Path = directory
        self.content_requests: list[tuple[str, Optional[str]]] = []
        self.failures_remaining: int = 0
        self.advertised_md5_overrides: dict[str, str] = {}
        self.response_delay_seconds: float = 0.0
        self.lock: threading.Lock = threading.Lock()
        outer: FakeZenodoServer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *arguments: Any) -> None:
                return

            def do_GET(self) -> None:
                if self.path == f"/api/records/{RECORD_ID}":
                    self._send_record()
                else:
                    self._send_content()

            def _send_record(self) -> None:
                files: list[dict[str, Any]] = []
                for name, content in FILE_CONTENTS.items():
                    md5: str = outer.advertised_md5_overrides.get(name, hashlib.md5(content).hexdigest())
                    files.append(
                        {
                            "key": name,
                            "size": len(content),
                            "checksum": f"md5:{md5}",
                            "links": {"self": f"{outer.base_url}/content/{name}"},
                        }
                    )
                body: bytes = json.dumps({"files": files}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_content(self) -> None:
                name: str = self.path.rsplit("/", 1)[-1]
                range_header: Optional[str] = self.headers.get("Range")
                with outer.lock:
                    outer.content_requests.append((name, range_header))
                    should_fail: bool = outer.failures_remaining > 0
                    outer.failures_remaining -= 1 if should_fail else 0
                if should_fail:
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

                    return

                content: bytes = FILE_CONTENTS[name]
                start_byte: int = 0
                if range_header:
                    start_byte = int(re.match(r"bytes=(\d+)-", range_header).group(1))
                    self.send_response(206)
                else:
                    self.send_response(200)
                body: bytes = content[start_byte:]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd: ThreadingHTTPServer = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url: str = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.api_url_template: str = self.base_url + "/api/records/{record_id}"
        self.thread: threading.Thread = threading.Thread(target=lambda: self.httpd.serve_forever(poll_interval=0.01), daemon=True)

    def content_request_count(self, name: str) -> int:
        return sum(1 for requested_name, _ in self.content_requests if requested_name == name)


@pytest.fixture
def server(tmp_path: Path) -> Iterator[FakeZenodoServer]:
    fake_server: FakeZenodoServer = FakeZenodoServer(tmp_path)
    fake_server.thread.start()
    yield fake_server
    fake_server.httpd.shutdown()
    fake_server.httpd.server_close()


@pytest.fixture
def fake_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        DATASET_SOURCES,
        "fake",
        download_module.DatasetSource(name="fake", record_id=RECORD_ID, default_file_names=("alpha.bin", "beta.bin")),
    )


def md5_of(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


def remote_file_for(server: FakeZenodoServer, name: str) -> RemoteFile:
    return RemoteFile(name, len(FILE_CONTENTS[name]), md5_of(FILE_CONTENTS[name]), f"{server.base_url}/content/{name}")


def test_fetch_record_files_parses_api(server: FakeZenodoServer) -> None:
    remote_files: list[RemoteFile] = fetch_record_files(RECORD_ID, server.api_url_template)

    assert [remote.name for remote in remote_files] == list(FILE_CONTENTS)
    assert remote_files[0].size_bytes == len(FILE_CONTENTS["alpha.bin"])
    assert remote_files[0].md5 == md5_of(FILE_CONTENTS["alpha.bin"])
    assert remote_files[0].url == f"{server.base_url}/content/alpha.bin"


def test_fetch_record_files_unreachable_raises() -> None:
    with pytest.raises(DownloadError):
        fetch_record_files(RECORD_ID, "http://127.0.0.1:1/api/{record_id}", timeout_seconds=1.0)


def test_download_file_success(server: FakeZenodoServer, tmp_path: Path) -> None:
    target_directory: Path = tmp_path / "out"
    target_directory.mkdir()

    final_path: Path = download_file(remote_file_for(server, "alpha.bin"), target_directory)

    assert final_path.read_bytes() == FILE_CONTENTS["alpha.bin"]
    assert not (target_directory / "alpha.bin.part").exists()


def test_checksum_mismatch_leaves_no_files(server: FakeZenodoServer, tmp_path: Path) -> None:
    target_directory: Path = tmp_path / "out"
    target_directory.mkdir()
    corrupt_file: RemoteFile = RemoteFile("alpha.bin", len(FILE_CONTENTS["alpha.bin"]), "0" * 32, f"{server.base_url}/content/alpha.bin")

    with pytest.raises(ChecksumMismatchError, match="alpha.bin"):
        download_file(corrupt_file, target_directory)

    assert not (target_directory / "alpha.bin").exists()
    assert not (target_directory / "alpha.bin.part").exists()


def test_complete_files_are_skipped(server: FakeZenodoServer, fake_dataset: None, tmp_path: Path) -> None:
    target_directory: Path = tmp_path / "out"
    keyword_arguments: dict[str, Any] = {"api_url_template": server.api_url_template, "backoff_seconds": 0.0}

    download_dataset("fake", target_directory, **keyword_arguments)
    download_dataset("fake", target_directory, **keyword_arguments)

    assert server.content_request_count("alpha.bin") == 1
    assert server.content_request_count("beta.bin") == 1


def test_verify_existing_redownloads_corrupt_file(server: FakeZenodoServer, fake_dataset: None, tmp_path: Path) -> None:
    target_directory: Path = tmp_path / "out"
    keyword_arguments: dict[str, Any] = {"api_url_template": server.api_url_template, "backoff_seconds": 0.0}
    download_dataset("fake", target_directory, **keyword_arguments)
    corrupted_content: bytes = b"x" * len(FILE_CONTENTS["alpha.bin"])
    (target_directory / "alpha.bin").write_bytes(corrupted_content)

    download_dataset("fake", target_directory, **keyword_arguments)
    assert (target_directory / "alpha.bin").read_bytes() == corrupted_content

    download_dataset("fake", target_directory, verify_existing=True, **keyword_arguments)
    assert (target_directory / "alpha.bin").read_bytes() == FILE_CONTENTS["alpha.bin"]
    assert server.content_request_count("alpha.bin") == 2


def test_resume_from_partial_file(server: FakeZenodoServer, tmp_path: Path) -> None:
    target_directory: Path = tmp_path / "out"
    target_directory.mkdir()
    (target_directory / "alpha.bin.part").write_bytes(FILE_CONTENTS["alpha.bin"][:1000])

    final_path: Path = download_file(remote_file_for(server, "alpha.bin"), target_directory)

    assert server.content_requests == [("alpha.bin", "bytes=1000-")]
    assert md5_of(final_path.read_bytes()) == md5_of(FILE_CONTENTS["alpha.bin"])


def test_retry_after_transient_error(server: FakeZenodoServer, tmp_path: Path) -> None:
    target_directory: Path = tmp_path / "out"
    target_directory.mkdir()
    server.failures_remaining = 2

    final_path: Path = download_file(remote_file_for(server, "alpha.bin"), target_directory, backoff_seconds=0.0)

    assert final_path.read_bytes() == FILE_CONTENTS["alpha.bin"]
    assert server.content_request_count("alpha.bin") == 3


def test_retries_exhausted(server: FakeZenodoServer, tmp_path: Path) -> None:
    server.failures_remaining = 10

    with pytest.raises(DownloadError, match="Giving up"):
        download_file(remote_file_for(server, "alpha.bin"), tmp_path, max_retries=1, backoff_seconds=0.0)

    assert server.content_request_count("alpha.bin") == 2


def test_insufficient_disk_space(
    server: FakeZenodoServer, fake_dataset: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(download_module.shutil, "disk_usage", lambda path: type("Usage", (), {"free": 10})())

    with pytest.raises(InsufficientDiskSpaceError) as error_info:
        download_dataset("fake", tmp_path / "out", api_url_template=server.api_url_template)

    needed_bytes: int = len(FILE_CONTENTS["alpha.bin"]) + len(FILE_CONTENTS["beta.bin"])
    assert str(needed_bytes) in str(error_info.value)
    assert "10 bytes" in str(error_info.value)
    assert server.content_requests == []


def test_unknown_file_name_lists_valid_names(server: FakeZenodoServer, fake_dataset: None, tmp_path: Path) -> None:
    with pytest.raises(UnknownFileError) as error_info:
        download_dataset("fake", tmp_path / "out", file_names=["nope.bin"], api_url_template=server.api_url_template)

    assert "nope.bin" in str(error_info.value)
    assert "alpha.bin" in str(error_info.value)


def test_selected_file_outside_defaults(server: FakeZenodoServer, fake_dataset: None, tmp_path: Path) -> None:
    local_paths: list[Path] = download_dataset(
        "fake", tmp_path / "out", file_names=["gamma.bin"], api_url_template=server.api_url_template
    )

    assert [path.name for path in local_paths] == ["gamma.bin"]
    assert server.content_request_count("alpha.bin") == 0


def test_duplicate_md5_warning(
    server: FakeZenodoServer, fake_dataset: None, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=download_module.logger.name):
        download_dataset(
            "fake", tmp_path / "out", file_names=["beta.bin", "gamma.bin"], api_url_template=server.api_url_template
        )

    assert any("identical md5" in record.message for record in caplog.records)
    assert server.content_request_count("beta.bin") == 1
    assert server.content_request_count("gamma.bin") == 1


def test_concurrent_callers_download_once(server: FakeZenodoServer, fake_dataset: None, tmp_path: Path) -> None:
    target_directory: Path = tmp_path / "out"
    errors: list[BaseException] = []

    def run_download() -> None:
        try:
            download_dataset("fake", target_directory, api_url_template=server.api_url_template)
        except BaseException as error:
            errors.append(error)

    threads: list[threading.Thread] = [threading.Thread(target=run_download) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert server.content_request_count("alpha.bin") == 1
    assert server.content_request_count("beta.bin") == 1


def test_default_data_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRATCH", "/scratch/user")
    assert default_data_directory("lsst") == Path("/scratch/user/astrouda_data/lsst")

    monkeypatch.delenv("SCRATCH")
    assert default_data_directory("gz2") == Path("data/gz2")


def test_default_file_lists() -> None:
    assert len(DATASET_SOURCES["lsst"].default_file_names) == 9
    assert "images_Y10_valid.npy" in DATASET_SOURCES["lsst"].default_file_names
    assert not any("150" in name for name in DATASET_SOURCES["lsst"].default_file_names)
    assert len(DATASET_SOURCES["gz2"].default_file_names) == 4


def test_cli_registration() -> None:
    arguments = build_argument_parser().parse_args(
        ["download", "--dataset", "lsst", "--file", "a.npy", "--file", "b.npy", "--verify-existing"]
    )

    assert arguments.command == "download"
    assert arguments.file_names == ["a.npy", "b.npy"]
    assert arguments.verify_existing is True
    assert arguments.directory is None


def test_cli_exit_codes(
    server: FakeZenodoServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setitem(
        DATASET_SOURCES,
        "lsst",
        download_module.DatasetSource(name="lsst", record_id=RECORD_ID, default_file_names=("alpha.bin",)),
    )
    monkeypatch.setattr(
        cli_registration_module,
        "download_dataset",
        functools.partial(download_dataset, api_url_template=server.api_url_template, backoff_seconds=0.0),
    )
    target_directory: Path = tmp_path / "out"

    assert main(["download", "--dataset", "lsst", "--directory", str(target_directory)]) == 0
    assert str(target_directory.resolve()) in capsys.readouterr().out
    assert (target_directory / "alpha.bin").exists()

    assert main(["download", "--dataset", "lsst", "--directory", str(target_directory), "--file", "missing.bin"]) == 1
    assert "missing.bin" in capsys.readouterr().out

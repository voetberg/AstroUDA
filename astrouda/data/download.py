import fcntl
import hashlib
import json
import logging
import os
import shutil
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

logger: logging.Logger = logging.getLogger(__name__)

ZENODO_API_URL_TEMPLATE: str = "https://zenodo.org/api/records/{record_id}"
LOCK_FILE_NAME: str = ".download.lock"
PART_SUFFIX: str = ".part"
CHUNK_SIZE_BYTES: int = 1024 * 1024
PROGRESS_INTERVAL_BYTES: int = 256 * 1024 * 1024
HTTP_PARTIAL_CONTENT: int = 206
HTTP_RANGE_NOT_SATISFIABLE: int = 416


class DownloadError(Exception):
    "Base class for every expected download failure."


class ChecksumMismatchError(DownloadError):
    "The finished file does not match the md5 published in the record."


class InsufficientDiskSpaceError(DownloadError):
    "Not enough free space in the target directory."


class UnknownFileError(DownloadError):
    "A requested file name is not in the record."


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size_bytes: int
    md5: str
    url: str


@dataclass(frozen=True)
class DatasetSource:
    name: str
    record_id: str
    default_file_names: tuple[str, ...]


DATASET_SOURCES: dict[str, DatasetSource] = {
    "lsst": DatasetSource(
        name="lsst",
        record_id="5514180",
        default_file_names=tuple(
            [f"images_{tag}_{part}.npy" for tag in ("Y1", "Y10") for part in ("train", "valid", "test")]
            + [f"labels_{part}.npy" for part in ("train", "valid", "test")]
        ),
    ),
    "gz2": DatasetSource(
        name="gz2",
        record_id="7473597",
        default_file_names=("decals.h5", "sdss_1.h5", "sdss_2.h5", "sdss_stripe82.h5"),
    ),
}


def default_data_directory(dataset_name: str) -> Path:
    "$SCRATCH/astrouda_data/<dataset> when SCRATCH is set, else ./data/<dataset>."
    scratch_directory: Optional[str] = os.environ.get("SCRATCH")
    if scratch_directory:
        return Path(scratch_directory) / "astrouda_data" / dataset_name

    return Path("data") / dataset_name


def fetch_record_files(
    record_id: str,
    api_url_template: str = ZENODO_API_URL_TEMPLATE,
    timeout_seconds: float = 30.0,
) -> list[RemoteFile]:
    record_url: str = api_url_template.format(record_id=record_id)
    logger.debug(f"Fetching record metadata from {record_url}")

    try:
        with urllib.request.urlopen(record_url, timeout=timeout_seconds) as response:
            record: dict = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise DownloadError(f"Could not fetch record metadata from {record_url}: {error}") from error

    remote_files: list[RemoteFile] = []
    for file_entry in record.get("files", []):
        checksum: str = file_entry["checksum"]
        algorithm, _, digest = checksum.partition(":")
        if algorithm != "md5":
            raise DownloadError(f"Unsupported checksum '{checksum}' for {file_entry['key']}, expected md5")

        remote_files.append(
            RemoteFile(
                name=file_entry["key"],
                size_bytes=int(file_entry["size"]),
                md5=digest,
                url=file_entry["links"]["self"],
            )
        )
    logger.debug(f"Record {record_id} lists {len(remote_files)} files: {[remote.name for remote in remote_files]}")

    return remote_files


def compute_md5(file_path: Path) -> str:
    digest = hashlib.md5()
    with open(file_path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _stream_to_part_file(
    remote_file: RemoteFile,
    part_path: Path,
    timeout_seconds: float,
    progress_interval_bytes: int,
) -> None:
    resume_offset: int = part_path.stat().st_size if part_path.exists() else 0
    if resume_offset > remote_file.size_bytes:
        logger.warning(f"Partial file {part_path} is larger than the remote file, restarting")
        part_path.unlink()
        resume_offset = 0
    if resume_offset == remote_file.size_bytes:
        logger.debug(f"Partial file {part_path} already has the full size, skipping transfer")

        return

    request: urllib.request.Request = urllib.request.Request(remote_file.url)
    if resume_offset > 0:
        request.add_header("Range", f"bytes={resume_offset}-")
        logger.info(f"Resuming {remote_file.name} at byte {resume_offset}")

    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        if resume_offset > 0 and response.status != HTTP_PARTIAL_CONTENT:
            logger.warning(f"Server ignored the Range header for {remote_file.name}, restarting from zero")
            resume_offset = 0

        bytes_written: int = resume_offset
        next_progress_report: int = bytes_written + progress_interval_bytes
        with open(part_path, "ab" if resume_offset > 0 else "wb") as part_handle:
            for chunk in iter(lambda: response.read(CHUNK_SIZE_BYTES), b""):
                part_handle.write(chunk)
                bytes_written += len(chunk)
                if bytes_written >= next_progress_report:
                    logger.info(f"{remote_file.name}: {bytes_written / 1e9:.2f} of {remote_file.size_bytes / 1e9:.2f} GB")
                    next_progress_report = bytes_written + progress_interval_bytes

    logger.debug(f"Transferred {remote_file.name} up to {bytes_written} bytes")


def download_file(
    remote_file: RemoteFile,
    directory: Path,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
    timeout_seconds: float = 60.0,
    progress_interval_bytes: int = PROGRESS_INTERVAL_BYTES,
) -> Path:
    "Streams to <name>.part (resuming a partial file), verifies the md5, then renames atomically."
    final_path: Path = directory / remote_file.name
    part_path: Path = directory / f"{remote_file.name}{PART_SUFFIX}"
    last_error: Optional[Exception] = None

    for attempt_index in range(max_retries + 1):
        if attempt_index > 0:
            sleep_seconds: float = backoff_seconds * 2 ** (attempt_index - 1)
            logger.warning(
                f"Retry {attempt_index}/{max_retries} for {remote_file.name} in {sleep_seconds:.1f}s after: {last_error}"
            )
            time.sleep(sleep_seconds)

        try:
            logger.info(f"Downloading {remote_file.name} ({remote_file.size_bytes / 1e9:.2f} GB), attempt {attempt_index + 1}")
            _stream_to_part_file(remote_file, part_path, timeout_seconds, progress_interval_bytes)
            break
        except urllib.error.HTTPError as error:
            if error.code == HTTP_RANGE_NOT_SATISFIABLE:
                part_path.unlink(missing_ok=True)
            elif error.code < 500 and error.code != 429:
                raise DownloadError(f"HTTP {error.code} for {remote_file.url}") from error
            last_error = error
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as error:
            last_error = error
    else:
        raise DownloadError(f"Giving up on {remote_file.name} after {max_retries + 1} attempts: {last_error}") from last_error

    actual_size_bytes: int = part_path.stat().st_size
    if actual_size_bytes != remote_file.size_bytes:
        part_path.unlink()
        raise DownloadError(f"{remote_file.name} has {actual_size_bytes} bytes, expected {remote_file.size_bytes}")

    actual_md5: str = compute_md5(part_path)
    if actual_md5 != remote_file.md5:
        part_path.unlink()
        logger.debug(f"Removed {part_path} after md5 mismatch")
        raise ChecksumMismatchError(f"{remote_file.name}: md5 {actual_md5} does not match expected {remote_file.md5}")

    os.replace(part_path, final_path)
    logger.info(f"Finished {final_path}, md5 verified")

    return final_path


def _select_files(
    dataset_source: DatasetSource,
    record_files: Sequence[RemoteFile],
    file_names: Optional[Sequence[str]],
) -> list[RemoteFile]:
    files_by_name: dict[str, RemoteFile] = {remote.name: remote for remote in record_files}
    requested_names: Sequence[str] = file_names if file_names else dataset_source.default_file_names

    unknown_names: list[str] = [name for name in requested_names if name not in files_by_name]
    if unknown_names:
        raise UnknownFileError(
            f"Unknown file(s) {unknown_names} for dataset '{dataset_source.name}'. Valid names: {sorted(files_by_name)}"
        )

    return [files_by_name[name] for name in dict.fromkeys(requested_names)]


def _warn_on_duplicate_md5(selected_files: Sequence[RemoteFile]) -> None:
    names_by_md5: dict[str, list[str]] = {}
    for remote_file in selected_files:
        names_by_md5.setdefault(remote_file.md5, []).append(remote_file.name)

    for md5, names in names_by_md5.items():
        if len(names) > 1:
            logger.warning(f"Files {names} have the identical md5 {md5}, their content is the same")


def download_dataset(
    dataset_name: str,
    directory: Path,
    file_names: Optional[Sequence[str]] = None,
    verify_existing: bool = False,
    api_url_template: str = ZENODO_API_URL_TEMPLATE,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
    timeout_seconds: float = 60.0,
) -> list[Path]:
    "Returns the local paths of the selected files. Safe to call from parallel processes sharing the directory."
    if dataset_name not in DATASET_SOURCES:
        raise DownloadError(f"Unknown dataset '{dataset_name}', choose from {sorted(DATASET_SOURCES)}")

    dataset_source: DatasetSource = DATASET_SOURCES[dataset_name]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    record_files: list[RemoteFile] = fetch_record_files(dataset_source.record_id, api_url_template, timeout_seconds)
    selected_files: list[RemoteFile] = _select_files(dataset_source, record_files, file_names)
    logger.debug(f"Selected {[remote.name for remote in selected_files]} for '{dataset_name}' in {directory}")

    local_paths: list[Path] = []
    with open(directory / LOCK_FILE_NAME, "w") as lock_handle:
        logger.debug(f"Waiting for lock {directory / LOCK_FILE_NAME}")
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        logger.debug("Lock acquired")

        try:
            pending_files: list[RemoteFile] = []
            for remote_file in selected_files:
                final_path: Path = directory / remote_file.name
                if final_path.exists() and final_path.stat().st_size == remote_file.size_bytes:
                    if not verify_existing or compute_md5(final_path) == remote_file.md5:
                        logger.info(f"Skipping {remote_file.name}, already complete")
                        continue
                    logger.warning(f"{final_path} fails md5 verification, downloading again")
                pending_files.append(remote_file)

            needed_bytes: int = 0
            for remote_file in pending_files:
                part_path: Path = directory / f"{remote_file.name}{PART_SUFFIX}"
                already_downloaded_bytes: int = part_path.stat().st_size if part_path.exists() else 0
                needed_bytes += max(remote_file.size_bytes - already_downloaded_bytes, 0)

            available_bytes: int = shutil.disk_usage(directory).free
            logger.debug(f"Disk space: need {needed_bytes} bytes, available {available_bytes} bytes")
            if needed_bytes > available_bytes:
                raise InsufficientDiskSpaceError(
                    f"Need {needed_bytes} bytes ({needed_bytes / 1e9:.2f} GB) in {directory} "
                    f"but only {available_bytes} bytes ({available_bytes / 1e9:.2f} GB) are available"
                )

            for remote_file in pending_files:
                download_file(remote_file, directory, max_retries, backoff_seconds, timeout_seconds)

            local_paths = [directory / remote_file.name for remote_file in selected_files]
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
            logger.debug("Lock released")

    _warn_on_duplicate_md5(selected_files)

    return local_paths

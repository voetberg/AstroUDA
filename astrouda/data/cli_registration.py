import argparse
import logging
from pathlib import Path
from typing import Any

from astrouda.data.download import DATASET_SOURCES, DownloadError, default_data_directory, download_dataset

logger: logging.Logger = logging.getLogger(__name__)


def download_command(arguments: argparse.Namespace) -> int:
    "Returns 0 on success, 1 on a download failure."
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    directory: Path = Path(arguments.directory) if arguments.directory else default_data_directory(arguments.dataset)

    try:
        local_paths: list[Path] = download_dataset(
            arguments.dataset,
            directory,
            file_names=arguments.file_names or None,
            verify_existing=arguments.verify_existing,
        )
    except DownloadError as error:
        logger.error(f"Download failed: {error}")
        print(f"Download failed: {error}")

        return 1

    logger.debug(f"Downloaded or verified {len(local_paths)} files")
    print(f"Use as data_directory: {directory.resolve()}")

    return 0


def register_data_subcommands(subparsers: Any) -> None:
    download_parser: argparse.ArgumentParser = subparsers.add_parser(
        "download", help="Download a dataset from Zenodo into a scratch directory"
    )
    download_parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SOURCES))
    download_parser.add_argument("--directory", default=None, help="Target directory, default $SCRATCH/astrouda_data/<dataset>")
    download_parser.add_argument(
        "--file", dest="file_names", action="append", default=[], metavar="NAME", help="Only this file (repeatable)"
    )
    download_parser.add_argument(
        "--verify-existing", action="store_true", help="Re-check the md5 of files that are already complete"
    )
    download_parser.set_defaults(handler=download_command)

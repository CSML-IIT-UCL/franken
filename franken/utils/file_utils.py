import hashlib
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
import requests
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def compute_md5(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(
            lambda: f.read(8192), b""
        ):  # Read in chunks to handle large files
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def validate_file(
    filename: Path, expected_size: int | None = None, expected_md5: str | None = None
) -> bool:
    if expected_md5 is not None:
        actual_md5 = compute_md5(filename)
        if actual_md5 != expected_md5:
            return False
    if expected_size is not None:
        actual_size = filename.stat().st_size
        if expected_size != actual_size:
            return False
    return True


def download_file(
    url: str,
    filename: Path,
    expected_size: int | None = None,
    expected_md5: str | None = None,
    desc: str | None = None,
):
    """Download atomically, retrying transient failures up to three times."""
    if (expected_md5 is not None or expected_size is not None) and filename.is_file():
        # Check that the file is correct to avoid re-download
        if validate_file(filename, expected_size, expected_md5):
            return filename

    for attempt in range(3):
        temporary = None
        try:
            with requests.get(url, stream=True, timeout=(10, 60)) as response:
                response.raise_for_status()
                data_size = 0
                data_md5 = hashlib.md5()
                with (
                    NamedTemporaryFile(
                        dir=filename.parent,
                        prefix=filename.name + ".",
                        suffix=".temp",
                        delete=False,
                    ) as file,
                    tqdm(
                        desc=desc or str(filename),
                        total=int(response.headers.get("content-length", 0)),
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar,
                ):
                    temporary = Path(file.name)
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                        data_md5.update(chunk)
                        data_size += len(chunk)
                        bar.update(len(chunk))
                if expected_size is not None and expected_size != data_size:
                    raise IOError("Incorrect file size", filename)
                if expected_md5 is not None and data_md5.hexdigest() != expected_md5:
                    raise IOError("Incorrect file MD5", filename)
                temporary.replace(filename)
                return filename
        except requests.RequestException as error:
            if isinstance(error, requests.HTTPError):
                retryable = (
                    error.response is not None
                    and error.response.status_code in {429, 500, 502, 503, 504}
                )
            else:
                retryable = isinstance(
                    error,
                    (
                        requests.ConnectionError,
                        requests.Timeout,
                        requests.exceptions.ChunkedEncodingError,
                    ),
                )
            if not retryable or attempt == 2:
                raise
            logger.warning(
                "Download attempt %d/3 failed for %s: %s", attempt + 1, url, error
            )
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        time.sleep(2**attempt)

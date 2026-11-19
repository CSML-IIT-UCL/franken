import hashlib
from io import BytesIO
from unittest.mock import patch

import pytest
import requests

from franken.utils.file_utils import download_file


def response(status=200, data=b"checkpoint"):
    result = requests.Response()
    result.status_code = status
    result.url = "https://example.test/model"
    result.raw = BytesIO(data)
    result.headers["content-length"] = str(len(data))
    return result


@pytest.mark.parametrize(
    "failure", [response(504), requests.Timeout(), requests.ConnectionError()]
)
def test_download_recovers_from_transient_errors(tmp_path, failure):
    destination = tmp_path / "model.pt"
    with (
        patch(
            "franken.utils.file_utils.requests.get", side_effect=[failure, response()]
        ) as get,
        patch("franken.utils.file_utils.time.sleep"),
    ):
        download_file("https://example.test/model", destination, expected_size=10)
    assert destination.read_bytes() == b"checkpoint"
    assert get.call_count == 2
    assert get.call_args.kwargs["timeout"] == (10, 60)
    assert not list(tmp_path.glob("*.temp"))


def test_download_does_not_retry_missing_models(tmp_path):
    with (
        patch(
            "franken.utils.file_utils.requests.get", return_value=response(404)
        ) as get,
        patch("franken.utils.file_utils.time.sleep") as sleep,
    ):
        with pytest.raises(requests.HTTPError):
            download_file("https://example.test/model", tmp_path / "model.pt")
    get.assert_called_once()
    sleep.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_download_retries_are_bounded(tmp_path):
    with (
        patch(
            "franken.utils.file_utils.requests.get",
            side_effect=[response(504) for _ in range(3)],
        ) as get,
        patch("franken.utils.file_utils.time.sleep"),
    ):
        with pytest.raises(requests.HTTPError):
            download_file("https://example.test/model", tmp_path / "model.pt")
    assert get.call_count == 3
    assert not list(tmp_path.iterdir())


def test_download_discards_interrupted_stream_before_retry(tmp_path):
    destination = tmp_path / "model.pt"
    interrupted = response()

    def broken_stream(chunk_size):
        yield b"partial"
        raise requests.exceptions.ChunkedEncodingError("connection lost")

    interrupted.iter_content = broken_stream

    responses = iter([interrupted, response()])

    def request(*args, **kwargs):
        assert not list(tmp_path.iterdir())
        return next(responses)

    with (
        patch("franken.utils.file_utils.requests.get", side_effect=request) as get,
        patch("franken.utils.file_utils.time.sleep"),
    ):
        download_file("https://example.test/model", destination)
    assert destination.read_bytes() == b"checkpoint"
    assert get.call_count == 2
    assert not list(tmp_path.glob("*.temp"))


@pytest.mark.parametrize("checks", [{"expected_size": 99}, {"expected_md5": "0" * 32}])
def test_failed_validation_preserves_destination(tmp_path, checks):
    destination = tmp_path / "model.pt"
    destination.write_bytes(b"old")
    with patch("franken.utils.file_utils.requests.get", return_value=response()) as get:
        with pytest.raises(IOError):
            download_file("https://example.test/model", destination, **checks)
    get.assert_called_once()
    assert destination.read_bytes() == b"old"
    assert not list(tmp_path.glob("*.temp"))


def test_valid_cached_download_does_not_make_request(tmp_path):
    destination = tmp_path / "model.pt"
    destination.write_bytes(b"checkpoint")
    with patch("franken.utils.file_utils.requests.get") as get:
        download_file(
            "https://example.test/model",
            destination,
            expected_md5=hashlib.md5(b"checkpoint").hexdigest(),
        )
    get.assert_not_called()

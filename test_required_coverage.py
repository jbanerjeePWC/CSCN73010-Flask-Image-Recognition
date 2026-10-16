"""Additional tests for the assignment's explicit coverage requirements.

The original test files are retained unchanged. These tests add focused checks
for the home route, a valid upload using mocked model functions, and invalid
image data without requiring a full model prediction.
"""

from io import BytesIO

import app as flask_app


def test_home_page_returns_successful_response(client):
    """The Flask home page can be accessed successfully."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.data


def test_valid_image_upload_returns_prediction(client, monkeypatch):
    """A valid image upload follows the preprocessing/prediction route."""
    calls = {"preprocess": 0, "predict": 0}

    def fake_preprocess(stream):
        calls["preprocess"] += 1
        assert stream.read(8) == b"\x89PNG\r\n\x1a\n"
        return "mock-preprocessed-image"

    def fake_predict(image):
        calls["predict"] += 1
        assert image == "mock-preprocessed-image"
        return 4

    monkeypatch.setattr(flask_app, "preprocess_img", fake_preprocess)
    monkeypatch.setattr(flask_app, "predict_result", fake_predict)

    # The route test mocks preprocessing, so only the PNG signature is needed.
    image_bytes = BytesIO(b"\x89PNG\r\n\x1a\nmock-image-data")

    response = client.post(
        "/prediction",
        data={"file": (image_bytes, "test.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Prediction" in response.data
    assert calls == {"preprocess": 1, "predict": 1}


def test_invalid_image_data_returns_error(client, monkeypatch):
    """An image-decoding error should be handled by the route."""

    def reject_image(_stream):
        raise ValueError("invalid image data")

    monkeypatch.setattr(flask_app, "preprocess_img", reject_image)

    response = client.post(
        "/prediction",
        data={"file": (BytesIO(b"not-an-image"), "broken.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data


def test_prediction_runtime_error_returns_error(client, monkeypatch):
    """A prediction runtime error should be handled gracefully."""

    def fake_preprocess(_stream):
        return "mock-preprocessed-image"

    def fail_prediction(_image):
        raise RuntimeError("simulated model runtime failure")

    monkeypatch.setattr(flask_app, "preprocess_img", fake_preprocess)
    monkeypatch.setattr(flask_app, "predict_result", fail_prediction)

    response = client.post(
        "/prediction",
        data={"file": (BytesIO(b"test-image-data"), "test.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data

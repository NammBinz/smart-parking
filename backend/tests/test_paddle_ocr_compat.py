import sys
from types import ModuleType

import numpy as np
import pytest

from app.ai import ocr_engines


IMAGE = np.zeros((40, 120, 3), dtype=np.uint8)


class FakeResult:
    json = {
        "res": {
            "rec_texts": ["29Z1", "58344"],
            "rec_scores": [0.95, 0.97],
        }
    }


class PredictReader:
    def __init__(self, output):
        self.output = output
        self.predict_calls = 0

    def predict(self, _image):
        self.predict_calls += 1
        return self.output

    def ocr(self, _image):
        raise AssertionError("The PaddleOCR 3.x path must not call ocr()")


def _recognize(monkeypatch, reader):
    monkeypatch.setattr(ocr_engines, "_get_paddle_reader", lambda: reader)
    return ocr_engines.recognize_with_paddleocr(IMAGE)


def test_paddle_3_result_object_combines_multiple_rows(monkeypatch):
    reader = PredictReader([FakeResult()])
    result = _recognize(monkeypatch, reader)
    assert result.raw_text == "29Z158344"
    assert result.normalized_text == "29Z158344"
    assert result.confidence == pytest.approx(0.96)
    assert reader.predict_calls == 1


def test_paddle_3_empty_result(monkeypatch):
    result = _recognize(monkeypatch, PredictReader([]))
    assert result.raw_text == ""
    assert result.normalized_text == ""
    assert result.confidence == 0.0
    assert result.is_valid is False


def test_paddle_3_single_dict_result(monkeypatch):
    reader = PredictReader(
        [{"res": {"rec_texts": ["51A4032"], "rec_scores": [0.91]}}]
    )
    result = _recognize(monkeypatch, reader)
    assert result.normalized_text == "51A4032"
    assert result.confidence == pytest.approx(0.91)


def test_paddle_3_multiple_result_objects_preserve_row_order(monkeypatch):
    class TopResult:
        json = {"res": {"rec_texts": ["29Z1"], "rec_scores": [0.95]}}

    class BottomResult:
        json = {"res": {"rec_texts": ["58344"], "rec_scores": [0.97]}}

    result = _recognize(monkeypatch, PredictReader([TopResult(), BottomResult()]))
    assert result.normalized_text == "29Z158344"
    assert result.confidence == pytest.approx(0.96)


@pytest.mark.parametrize(
    ("label", "plate"),
    (
        ("HONDA", "29Z138172"),
        ("HOND", "29S668708"),
        ("YAMAHA", "50G20995"),
        ("TOYOTA", "51D81343"),
        ("ABC", "71B158609"),
    ),
)
def test_non_plate_text_is_filtered_generically(monkeypatch, label, plate):
    reader = PredictReader(
        [
            {
                "res": {
                    "rec_texts": [label, plate],
                    "rec_scores": [0.99, 0.94],
                }
            }
        ]
    )
    result = _recognize(monkeypatch, reader)
    assert result.normalized_text == plate
    assert result.noise_detected is True
    assert tuple(fragment.text for fragment in result.fragments) == (plate,)


def test_legacy_nested_list_uses_ocr_without_cls_argument(monkeypatch):
    class LegacyReader:
        def ocr(self, _image):
            return [
                [
                    ([[0, 0], [50, 0], [50, 15], [0, 15]], ("29Z1", 0.95)),
                    ([[0, 20], [60, 20], [60, 40], [0, 40]], ("58344", 0.97)),
                ]
            ]

    result = _recognize(monkeypatch, LegacyReader())
    assert result.normalized_text == "29Z158344"
    assert result.confidence == pytest.approx(0.96)


def test_paddle_3_constructor_and_narrow_legacy_fallback(monkeypatch):
    calls = []

    class FakePaddleOCR:
        def __new__(cls, **kwargs):
            calls.append(kwargs)
            if "use_doc_orientation_classify" in kwargs:
                raise ValueError("Unknown argument: use_doc_orientation_classify")
            return object.__new__(cls)

    module = ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    ocr_engines._get_paddle_reader.cache_clear()
    try:
        ocr_engines._get_paddle_reader()
    finally:
        ocr_engines._get_paddle_reader.cache_clear()

    assert calls == [
        {
            "lang": "en",
            "device": "cpu",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "enable_mkldnn": False,
        },
        {"lang": "en"},
    ]


def test_paddle_3_constructor_uses_supported_options_without_fallback(monkeypatch):
    calls = []

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    module = ModuleType("paddleocr")
    module.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    ocr_engines._get_paddle_reader.cache_clear()
    try:
        ocr_engines._get_paddle_reader()
    finally:
        ocr_engines._get_paddle_reader.cache_clear()

    assert calls == [
        {
            "lang": "en",
            "device": "cpu",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "enable_mkldnn": False,
        }
    ]


def test_constructor_does_not_swallow_unrelated_model_errors(monkeypatch):
    class BrokenPaddleOCR:
        def __init__(self, **_kwargs):
            raise RuntimeError("model files are corrupt")

    module = ModuleType("paddleocr")
    module.PaddleOCR = BrokenPaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    ocr_engines._get_paddle_reader.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="model files are corrupt"):
            ocr_engines._get_paddle_reader()
    finally:
        ocr_engines._get_paddle_reader.cache_clear()


def test_constructor_does_not_treat_unrelated_unknown_argument_as_version_issue(
    monkeypatch,
):
    class BrokenPaddleOCR:
        def __init__(self, **_kwargs):
            raise ValueError("Unknown argument: model_precision")

    module = ModuleType("paddleocr")
    module.PaddleOCR = BrokenPaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    ocr_engines._get_paddle_reader.cache_clear()
    try:
        with pytest.raises(ValueError, match="model_precision"):
            ocr_engines._get_paddle_reader()
    finally:
        ocr_engines._get_paddle_reader.cache_clear()

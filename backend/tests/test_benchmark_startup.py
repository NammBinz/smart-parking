import ast
from pathlib import Path

from scripts import benchmark_recognition


def test_ocr_adapter_has_no_module_scope_paddle_import():
    source = (
        Path(__file__).parents[1] / "app" / "ai" / "ocr_engines.py"
    ).read_text(encoding="utf-8")
    module = ast.parse(source)
    top_level_imports = []
    for node in module.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.append(node.module)

    assert not any(
        name == "paddle" or name.startswith(("paddle.", "paddleocr", "paddlex"))
        for name in top_level_imports
    )


def test_paddle_initialization_happens_after_yolo_warmup():
    events = []

    def initialize_torch():
        events.append("torch")

    def analyze(_image, _confidence):
        assert events[-1] == "torch"
        events.append("yolo")

    def diagnostics():
        assert events == [
            "message:Initializing PyTorch/YOLO...",
            "torch",
            "yolo",
            "message:PyTorch/YOLO initialized",
        ]
        events.append("diagnostics")
        return {"PaddleOCR": "3.7.0"}

    def initialize_paddle():
        assert events[-2:] == ["diagnostics", "message:PaddleOCR: 3.7.0"]
        events.append("paddle")

    benchmark_recognition._initialize_benchmark_runtimes(
        object(),
        0.5,
        use_paddle=True,
        warmup=False,
        torch_initialize=initialize_torch,
        analyze=analyze,
        paddle_diagnostics=diagnostics,
        paddle_initialize=initialize_paddle,
        emit=lambda message: events.append(f"message:{message}"),
    )

    assert (
        events.index("torch")
        < events.index("yolo")
        < events.index("diagnostics")
        < events.index("paddle")
    )


def test_no_paddle_work_occurs_when_comparison_is_disabled():
    events = []
    benchmark_recognition._initialize_benchmark_runtimes(
        object(),
        0.5,
        use_paddle=False,
        warmup=False,
        torch_initialize=lambda: events.append("torch"),
        analyze=lambda *_args: events.append("yolo"),
        paddle_diagnostics=lambda: events.append("diagnostics"),
        paddle_initialize=lambda: events.append("paddle"),
        emit=lambda message: events.append(message),
    )
    assert events == []

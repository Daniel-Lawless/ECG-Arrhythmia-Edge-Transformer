import sys
from pathlib import Path

import pytest

from ecg_arrhythmia.deployment import quantize_onnx as quantize_module
from ecg_arrhythmia.deployment.quantize_onnx import (
    BYTES_PER_MIB,
    clean_value_info,
    file_size_mib,
    quantize_model,
    size_reduction_percentage,
    summarise_operator_changes,
    validate_strict_shape_inference,
)

FP32_COUNTS = {"Conv": 8, "MatMul": 12, "Gemm": 2, "Relu": 8, "Add": 20}
INT8_COUNTS = {
    "Conv": 8,
    "MatMulInteger": 12,
    "DynamicQuantizeLinear": 12,
    "Relu": 8,
    "Add": 20,
    "Gemm": 1,
}


# ---------------------------------------------------------------------
#                            Size Helpers
# ---------------------------------------------------------------------


def test_file_size_converts_bytes_to_mebibytes():
    assert file_size_mib(BYTES_PER_MIB) == pytest.approx(1.0)
    assert file_size_mib(0) == pytest.approx(0.0)


def test_a_negative_file_size_is_rejected():
    with pytest.raises(ValueError, match="must not be negative"):
        file_size_mib(-1)


def test_size_reduction_is_a_percentage_of_the_fp32_size():
    assert size_reduction_percentage(1000, 250) == pytest.approx(75.0)
    assert size_reduction_percentage(1000, 1000) == pytest.approx(0.0)


def test_a_grown_model_reports_a_negative_reduction():
    # Dynamic quantisation should shrink the file, but if it ever grows
    # the report must say so rather than clamp to zero.
    assert size_reduction_percentage(1000, 1100) == pytest.approx(-10.0)


def test_size_reduction_rejects_impossible_sizes():
    with pytest.raises(ValueError, match="must be positive"):
        size_reduction_percentage(0, 100)

    with pytest.raises(ValueError, match="must not be negative"):
        size_reduction_percentage(1000, -1)


# ---------------------------------------------------------------------
#                        Operator Summarisation
# ---------------------------------------------------------------------


def test_operator_changes_are_summarised_by_type():
    summary = summarise_operator_changes(FP32_COUNTS, INT8_COUNTS)

    assert summary["operator_types_introduced"] == [
        "DynamicQuantizeLinear",
        "MatMulInteger",
    ]
    assert summary["operator_types_removed"] == ["MatMul"]
    assert summary["operator_types_reduced"] == ["Gemm"]


def test_quantised_and_floating_point_operators_are_separated():
    summary = summarise_operator_changes(FP32_COUNTS, INT8_COUNTS)

    assert summary["quantised_operator_types_present"] == [
        "DynamicQuantizeLinear",
        "MatMulInteger",
    ]
    assert summary["remaining_floating_point_operator_types"] == [
        "Add",
        "Conv",
        "Gemm",
        "Relu",
    ]


def test_identical_graphs_summarise_to_no_changes():
    summary = summarise_operator_changes(FP32_COUNTS, dict(FP32_COUNTS))

    assert summary["operator_types_introduced"] == []
    assert summary["operator_types_removed"] == []
    assert summary["operator_types_reduced"] == []
    assert summary["quantised_operator_types_present"] == []


# ---------------------------------------------------------------------
#                       Metadata Cleaning Helpers
# ---------------------------------------------------------------------


def _tiny_model():
    """A minimal real ONNX model with value_info, an initializer and IO."""

    from onnx import TensorProto, helper

    weight = helper.make_tensor(
        name="weight",
        data_type=TensorProto.FLOAT,
        dims=[2, 2],
        vals=[1.0, 0.0, 0.0, 1.0],
    )
    node = helper.make_node("MatMul", inputs=["x", "weight"], outputs=["y"])
    graph = helper.make_graph(
        nodes=[node],
        name="tiny",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        initializer=[weight],
        value_info=[
            # The stale-metadata hazard in miniature: value_info naming
            # an initializer, exactly what the dynamo export produces.
            helper.make_tensor_value_info("weight", TensorProto.FLOAT, [2, 2]),
        ],
    )

    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_cleaning_removes_only_value_info(tmp_path):
    import onnx

    source_path = tmp_path / "model.onnx"
    cleaned_path = tmp_path / "model_cleaned.onnx"
    onnx.save(_tiny_model(), source_path)

    removed = clean_value_info(source_path, cleaned_path)
    cleaned = onnx.load(cleaned_path)

    assert removed == 1
    assert len(cleaned.graph.value_info) == 0

    # Everything that defines the computation survives untouched.
    assert [graph_input.name for graph_input in cleaned.graph.input] == ["x"]
    assert [graph_output.name for graph_output in cleaned.graph.output] == ["y"]
    assert [node.op_type for node in cleaned.graph.node] == ["MatMul"]
    assert [init.name for init in cleaned.graph.initializer] == ["weight"]

    # The source file itself is never modified.
    assert len(onnx.load(source_path).graph.value_info) == 1


def test_strict_shape_inference_accepts_a_consistent_model(tmp_path):
    import onnx

    model_path = tmp_path / "model.onnx"
    onnx.save(_tiny_model(), model_path)

    validate_strict_shape_inference(model_path)


# ---------------------------------------------------------------------
#                           Path Validation
# ---------------------------------------------------------------------


def test_the_fp32_source_cannot_be_overwritten(tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fp32")

    with pytest.raises(ValueError, match="must not overwrite the FP32 source"):
        quantize_model(
            input_model_path=model_path,
            output_model_path=model_path,
        )


def test_the_same_file_through_different_spellings_is_still_rejected(tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fp32")

    with pytest.raises(ValueError, match="must not overwrite the FP32 source"):
        quantize_model(
            input_model_path=model_path,
            output_model_path=tmp_path / "subdir" / ".." / "model.onnx",
        )


def test_a_missing_source_model_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="No FP32 ONNX model"):
        quantize_model(
            input_model_path=tmp_path / "missing.onnx",
            output_model_path=tmp_path / "model_int8.onnx",
        )


# ---------------------------------------------------------------------
#                     Quantisation Flow (Mocked)
# ---------------------------------------------------------------------


class _FakeQuantType:
    QInt8 = "QInt8"


@pytest.fixture
def mocked_quantisation(monkeypatch, tmp_path):
    """
    Replace the ONNX Runtime calls with fakes and record their arguments.

    The flow under test is orchestration: pre-processing feeds
    quantisation, validation runs on the output, and the report reflects
    what happened. The real quantiser is exercised by the CLI run.
    """

    calls: dict = {
        "quantize": [],
        "preprocess": [],
        "validated": [],
        "cleaned": [],
        "strict": [],
        # When set, the full pass raises the way symbolic shape inference
        # does on the dynamo-exported graph; only the retry succeeds.
        "fail_full_preprocess": False,
    }

    def fake_quant_pre_process(input_model_path, output_model_path, **kwargs):
        calls["preprocess"].append((input_model_path, output_model_path, kwargs))

        if calls["fail_full_preprocess"] and not kwargs.get("skip_symbolic_shape"):
            raise AssertionError("symbolic shape inference failed")

        Path(output_model_path).write_bytes(b"preprocessed-fp32")

    def fake_quantize_dynamic(model_input, model_output, **kwargs):
        calls["quantize"].append({"model_input": model_input, **kwargs})
        Path(model_output).write_bytes(b"int8")

    def fake_preprocess_module():
        pass

    fake_preprocess_module.quant_pre_process = fake_quant_pre_process

    class FakeQuantizationModule:
        QuantType = _FakeQuantType
        quantize_dynamic = staticmethod(fake_quantize_dynamic)

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime.quantization",
        FakeQuantizationModule,
    )
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime.quantization.shape_inference",
        fake_preprocess_module,
    )

    # Production validation helpers are reused by name; stub them to
    # record that they ran against the quantised output.
    monkeypatch.setattr(
        quantize_module,
        "validate_onnx_model",
        lambda path: calls["validated"].append(("structural", str(path))),
    )
    monkeypatch.setattr(
        quantize_module,
        "validate_strict_shape_inference",
        lambda path: calls["strict"].append(str(path)),
    )

    def fake_clean_value_info(source_path, cleaned_path):
        calls["cleaned"].append((str(source_path), str(cleaned_path)))
        Path(cleaned_path).write_bytes(b"cleaned-fp32")
        return 258

    monkeypatch.setattr(quantize_module, "clean_value_info", fake_clean_value_info)

    class FakeSession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(
        quantize_module,
        "create_onnx_session",
        lambda path: FakeSession(),
    )
    monkeypatch.setattr(
        quantize_module,
        "validate_session_contract",
        lambda session: calls["validated"].append(("contract", "session")),
    )
    monkeypatch.setattr(
        quantize_module,
        "run_smoke_inference",
        lambda path: True,
    )
    monkeypatch.setattr(
        quantize_module,
        "operator_type_counts",
        lambda path: (
            dict(FP32_COUNTS) if "int8" not in Path(path).name else dict(INT8_COUNTS)
        ),
    )

    fp32_path = tmp_path / "model.onnx"
    fp32_path.write_bytes(b"fp32-model-bytes")

    return calls, fp32_path, tmp_path / "model_int8.onnx"


def test_quantisation_uses_dynamic_qint8_configuration(mocked_quantisation):
    calls, fp32_path, int8_path = mocked_quantisation

    quantize_model(input_model_path=fp32_path, output_model_path=int8_path)

    [quantize_call] = calls["quantize"]
    assert quantize_call["weight_type"] == _FakeQuantType.QInt8


def test_the_quantiser_consumes_the_cleaned_preprocessed_model(mocked_quantisation):
    calls, fp32_path, int8_path = mocked_quantisation

    report = quantize_model(input_model_path=fp32_path, output_model_path=int8_path)

    # Pre-processing ran once, in full, on the FP32 source; cleaning
    # consumed its output; the quantiser consumed the cleaned copy.
    [(preprocess_input, preprocess_output, kwargs)] = calls["preprocess"]
    [(clean_source, clean_output)] = calls["cleaned"]
    [quantize_call] = calls["quantize"]

    assert Path(preprocess_input) == fp32_path
    assert "skip_symbolic_shape" not in kwargs
    assert Path(clean_source) == Path(preprocess_output)
    assert Path(quantize_call["model_input"]) == Path(clean_output)

    assert report["quantisation"]["preprocessing_used"] is True
    assert report["quantisation"]["preprocessing_mode"] == "full"
    assert report["quantisation"]["preprocessing_fallback_used"] is False
    assert report["quantisation"]["full_preprocessing_error"] is None
    assert report["quantisation"]["value_info_entries_removed"] == 258

    # Strict shape inference guarded both the source and the cleaned copy.
    assert calls["strict"] == [str(fp32_path), clean_output]

    # Both intermediates are removed unless explicitly kept.
    assert not Path(preprocess_output).exists()
    assert not Path(clean_output).exists()


def test_a_symbolic_shape_failure_retries_without_symbolic_inference(
    mocked_quantisation,
):
    calls, fp32_path, int8_path = mocked_quantisation
    calls["fail_full_preprocess"] = True

    report = quantize_model(input_model_path=fp32_path, output_model_path=int8_path)

    # The full pass failed, the retry skipped symbolic shape inference,
    # and cleaning consumed the retried output.
    [first, second] = calls["preprocess"]
    assert "skip_symbolic_shape" not in first[2]
    assert second[2] == {"skip_symbolic_shape": True}

    [(clean_source, clean_output)] = calls["cleaned"]
    [quantize_call] = calls["quantize"]
    assert Path(clean_source) == Path(second[1])
    assert Path(quantize_call["model_input"]) == Path(clean_output)

    assert report["quantisation"]["preprocessing_used"] is True
    assert report["quantisation"]["preprocessing_mode"] == "skip_symbolic_shape"
    assert report["quantisation"]["preprocessing_fallback_used"] is True

    # The original failure reason is preserved in the report.
    assert "AssertionError" in report["quantisation"]["full_preprocessing_error"]


def test_a_preprocessing_failure_falls_back_to_the_source_model(
    mocked_quantisation,
    monkeypatch,
):
    calls, fp32_path, int8_path = mocked_quantisation

    monkeypatch.setattr(
        quantize_module,
        "preprocess_model",
        lambda input_model_path, preprocessed_path: ("none", "AssertionError: x"),
    )

    report = quantize_model(input_model_path=fp32_path, output_model_path=int8_path)

    # Cleaning still runs, against the original source.
    [(clean_source, clean_output)] = calls["cleaned"]
    [quantize_call] = calls["quantize"]
    assert Path(clean_source) == Path(fp32_path)
    assert Path(quantize_call["model_input"]) == Path(clean_output)

    assert report["quantisation"]["preprocessing_used"] is False
    assert report["quantisation"]["preprocessing_mode"] == "none"
    assert report["quantisation"]["preprocessing_fallback_used"] is True
    assert report["quantisation"]["full_preprocessing_error"] == "AssertionError: x"


def test_intermediates_are_kept_on_request(mocked_quantisation):
    calls, fp32_path, int8_path = mocked_quantisation

    quantize_model(
        input_model_path=fp32_path,
        output_model_path=int8_path,
        keep_preprocessed_model=True,
    )

    [(_, preprocess_output, _)] = calls["preprocess"]
    [(_, clean_output)] = calls["cleaned"]
    assert Path(preprocess_output).exists()
    assert Path(clean_output).exists()


def test_intermediates_are_removed_even_when_quantisation_fails(
    mocked_quantisation,
    monkeypatch,
):
    calls, fp32_path, int8_path = mocked_quantisation

    def failing_quantize_dynamic(model_input, model_output, **kwargs):
        raise RuntimeError("quantisation exploded")

    sys.modules["onnxruntime.quantization"].quantize_dynamic = staticmethod(
        failing_quantize_dynamic
    )

    with pytest.raises(RuntimeError, match="quantisation exploded"):
        quantize_model(input_model_path=fp32_path, output_model_path=int8_path)

    [(_, preprocess_output, _)] = calls["preprocess"]
    [(_, clean_output)] = calls["cleaned"]
    assert not Path(preprocess_output).exists()
    assert not Path(clean_output).exists()


def test_the_report_covers_sizes_validation_and_operators(mocked_quantisation):
    calls, fp32_path, int8_path = mocked_quantisation

    report = quantize_model(input_model_path=fp32_path, output_model_path=int8_path)

    size = report["size"]
    assert size["fp32_bytes"] == len(b"fp32-model-bytes")
    assert size["int8_bytes"] == len(b"int8")
    assert size["reduction_bytes"] == size["fp32_bytes"] - size["int8_bytes"]
    assert size["reduction_percentage"] == pytest.approx(
        size_reduction_percentage(size["fp32_bytes"], size["int8_bytes"])
    )

    validation = report["validation"]
    assert validation["structural_validation_passed"] is True
    assert validation["contract_validation_passed"] is True
    assert validation["smoke_inference_passed"] is True
    assert validation["execution_providers"] == ["CPUExecutionProvider"]

    # The production validators ran against the quantised output.
    assert ("structural", str(int8_path)) in calls["validated"]
    assert ("contract", "session") in calls["validated"]

    operators = report["operators"]
    assert operators["fp32_operator_counts"] == FP32_COUNTS
    assert operators["int8_operator_counts"] == INT8_COUNTS
    assert operators["quantised_operator_types_present"] == [
        "DynamicQuantizeLinear",
        "MatMulInteger",
    ]

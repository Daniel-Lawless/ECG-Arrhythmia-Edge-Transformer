import pytest

from ecg_arrhythmia.transport.control_server import parse_args


@pytest.fixture(autouse=True)
def clear_destination_environment(monkeypatch):
    monkeypatch.delenv("ECG_DATA_HOST", raising=False)
    monkeypatch.delenv("ECG_DATA_PORT", raising=False)


def test_native_cli_defaults_are_preserved():
    args = parse_args(["--host", "192.0.2.10"])

    assert args.host == "192.0.2.10"
    assert args.port == 8765
    assert args.control_host == "0.0.0.0"
    assert args.control_port == 8767
    assert args.mode == "real_time"
    assert str(args.model_path).replace("\\", "/") == (
        "artifacts/models/ecg_sequence_transformer.onnx"
    )


def test_environment_configures_only_outbound_destination(monkeypatch):
    monkeypatch.setenv("ECG_DATA_HOST", "dashboard")
    monkeypatch.setenv("ECG_DATA_PORT", "9875")
    args = parse_args([])

    assert (args.host, args.port) == ("dashboard", 9875)
    assert (args.control_host, args.control_port) == ("0.0.0.0", 8767)


def test_explicit_cli_overrides_environment_including_invalid_port(monkeypatch):
    monkeypatch.setenv("ECG_DATA_HOST", "dashboard")
    monkeypatch.setenv("ECG_DATA_PORT", "invalid")
    args = parse_args(["--host", "192.0.2.20", "--port", "8888"])

    assert (args.host, args.port) == ("192.0.2.20", 8888)


@pytest.mark.parametrize("host", [None, "", "   "])
def test_missing_destination_is_a_clear_configuration_error(monkeypatch, capsys, host):
    if host is not None:
        monkeypatch.setenv("ECG_DATA_HOST", host)

    with pytest.raises(SystemExit) as error:
        parse_args([])

    assert error.value.code == 2
    assert "ECG_DATA_HOST" in capsys.readouterr().err


@pytest.mark.parametrize("port", ["invalid", "0", "-1", "65536"])
def test_invalid_environment_port_is_rejected(monkeypatch, port):
    monkeypatch.setenv("ECG_DATA_HOST", "dashboard")
    monkeypatch.setenv("ECG_DATA_PORT", port)

    with pytest.raises(SystemExit):
        parse_args([])

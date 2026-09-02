# syntax=docker/dockerfile:1

# Same target-platform base for builders and runtimes; no forced amd64 stage.
FROM python:3.12-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
# Required by both the Arrow (Streamlit) and CPU ONNX Runtime wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 ecg \
    && useradd --uid 10001 --gid ecg --create-home ecg
WORKDIR /app

# Only lightweight shared code enters both package builds.
FROM base AS shared-source
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ecg_arrhythmia/__init__.py ./src/ecg_arrhythmia/
COPY src/ecg_arrhythmia/data/__init__.py src/ecg_arrhythmia/data/label_mapping.py src/ecg_arrhythmia/data/mitdb_records.py ./src/ecg_arrhythmia/data/
COPY src/ecg_arrhythmia/transport/__init__.py src/ecg_arrhythmia/transport/protocol.py src/ecg_arrhythmia/transport/control_config.py src/ecg_arrhythmia/transport/control_protocol.py ./src/ecg_arrhythmia/transport/

FROM shared-source AS dashboard-build
COPY src/ecg_arrhythmia/dashboard/ ./src/ecg_arrhythmia/dashboard/
COPY src/ecg_arrhythmia/transport/tcp_receiver.py src/ecg_arrhythmia/transport/control_client.py ./src/ecg_arrhythmia/transport/
RUN python -m pip install --prefix=/install --only-binary=:all: ".[dashboard]"

FROM shared-source AS edge-build
COPY src/ecg_arrhythmia/data/load_record.py ./src/ecg_arrhythmia/data/
COPY src/ecg_arrhythmia/detection/__init__.py src/ecg_arrhythmia/detection/r_peak_detector.py src/ecg_arrhythmia/detection/xqrs_detector.py ./src/ecg_arrhythmia/detection/
COPY src/ecg_arrhythmia/preprocessing/__init__.py src/ecg_arrhythmia/preprocessing/beat_extraction.py ./src/ecg_arrhythmia/preprocessing/
COPY src/ecg_arrhythmia/streaming/ ./src/ecg_arrhythmia/streaming/
COPY src/ecg_arrhythmia/telemetry/ ./src/ecg_arrhythmia/telemetry/
COPY src/ecg_arrhythmia/transport/tcp_receiver.py src/ecg_arrhythmia/transport/tcp_sender.py src/ecg_arrhythmia/transport/send_record.py src/ecg_arrhythmia/transport/control_server.py ./src/ecg_arrhythmia/transport/
RUN python -m pip install --prefix=/install --only-binary=:all: ".[edge]"

FROM base AS dashboard
COPY --from=dashboard-build /install/ /usr/local/
COPY --from=dashboard-build /build/src/ecg_arrhythmia/dashboard/app.py /app/dashboard.py
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    ECG_DASHBOARD_HOST=0.0.0.0 \
    ECG_DASHBOARD_PORT=8765 \
    ECG_LIVE_HTTP_HOST=0.0.0.0 \
    ECG_LIVE_HTTP_PORT=8766 \
    ECG_LIVE_HTTP_PUBLIC_URL=http://127.0.0.1:8766 \
    ECG_PI_CONTROL_HOST=127.0.0.1 \
    ECG_PI_CONTROL_PORT=8767
RUN python -m pip check \
    && python -c "import importlib.util as u; assert all(u.find_spec(m) is None for m in ('torch', 'onnx', 'onnxruntime', 'wfdb', 'neurokit2', 'ecg_arrhythmia.streaming')); import ecg_arrhythmia.dashboard.live_ecg_component, ecg_arrhythmia.dashboard.record_control"
USER ecg
EXPOSE 8501 8765 8766
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2).read()"]
STOPSIGNAL SIGINT
CMD ["python", "-m", "streamlit", "run", "/app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

FROM base AS edge
COPY --from=edge-build /install/ /usr/local/
COPY artifacts/models/ecg_sequence_transformer.onnx /app/artifacts/models/ecg_sequence_transformer.onnx
ENV ECG_DATA_PORT=8765 \
    MPLCONFIGDIR=/tmp/matplotlib
RUN python -m pip check \
    && python -c "import importlib.util as u; assert all(u.find_spec(m) is None for m in ('torch', 'onnx', 'neurokit2', 'streamlit', 'plotly', 'ecg_arrhythmia.dashboard')); import ecg_arrhythmia.transport.control_server" \
    && python -c "from pathlib import Path; from ecg_arrhythmia.streaming.onnx_sequence_classifier import ONNXSequenceClassifier; ONNXSequenceClassifier(Path('artifacts/models/ecg_sequence_transformer.onnx'))"
USER ecg
EXPOSE 8767
# Let the existing KeyboardInterrupt/finally path stop and join the worker.
STOPSIGNAL SIGINT
CMD ["python", "-m", "ecg_arrhythmia.transport.control_server"]

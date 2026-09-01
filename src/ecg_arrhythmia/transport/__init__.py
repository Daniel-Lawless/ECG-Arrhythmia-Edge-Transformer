"""
Pi -> PC streaming transport (Section 6.1).

A deliberately small transport layer: a versioned newline-delimited
JSON protocol (`protocol`), a persistent TCP sender (`tcp_sender`), a
framing-correct development receiver (`tcp_receiver`) and a demo entry
point that streams one record through the existing FP32 inference
pipeline (`send_record`).

`protocol` and the two TCP modules import only the standard library, so
the receiver can run on a machine without the project's scientific
dependencies installed.
"""

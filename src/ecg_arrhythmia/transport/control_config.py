# The Pi's control listener port. Distinct from the ECG data port
# (8765, Pi -> PC) and from the dashboard's loopback live endpoint
# (8766, PC-internal) so all three are unambiguous in logs and docs.
DEFAULT_CONTROL_PORT = 8767

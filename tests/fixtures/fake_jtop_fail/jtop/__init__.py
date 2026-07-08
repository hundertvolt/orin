"""
Fake `jtop` package whose context manager fails to open - simulates the
jetson_stats service not running. Point PYTHONPATH at this directory's
parent to use it.
"""
import time


class JtopException(Exception):
    pass


class jtop:
    def __enter__(self):
        # A real jtop() failure isn't instant either - it attempts a
        # socket/IPC connection to the jetson_stats service and only
        # raises after that fails. The small delay here is realistic
        # *and* keeps this deterministic: it gives the (local, fast)
        # MQTT CONNACK round-trip time to land before the main loop
        # dies, instead of racing it.
        time.sleep(0.3)
        raise JtopException("jetson_stats service is not running (simulated)")

    def __exit__(self, exc_type, exc, tb):
        return False

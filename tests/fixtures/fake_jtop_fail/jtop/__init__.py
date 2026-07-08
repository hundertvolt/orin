"""
Fake `jtop` package whose context manager fails to open - simulates the
jetson_stats service not running. Point PYTHONPATH at this directory's
parent to use it.

Deliberately instant: mqtt_telemetry.py now waits for the MQTT
connection to settle before ever attempting jtop(), so this no longer
needs an artificial delay to avoid racing the CONNACK - proving that
fix actually closes the race rather than relying on timing luck.
"""


class JtopException(Exception):
    pass


class jtop:
    def __enter__(self):
        raise JtopException("jetson_stats service is not running (simulated)")

    def __exit__(self, exc_type, exc, tb):
        return False

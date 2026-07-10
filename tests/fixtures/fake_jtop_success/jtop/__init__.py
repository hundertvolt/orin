"""
Minimal stand-in for the `jtop` package (jetson-stats), for running
mqtt_telemetry.py as a real subprocess in integration tests without
actual Jetson hardware or the jetson_stats service. Point PYTHONPATH at
this directory's parent to use it.
"""
import datetime


class jtop:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def ok(self, spin=False):
        return True

    def close(self):
        pass

    @property
    def stats(self):
        return {
            "CPU1": 5.0, "CPU2": 6.0, "CPU3": 4.0, "CPU4": 7.0,
            "Temp cpu": 45.0, "Temp gpu": 44.0,
            "RAM": 0.3, "SWAP": 0.0, "GPU": 2.0,
            "uptime": datetime.timedelta(seconds=120),
            "Power TOT": 4500,
        }

    @property
    def fan(self):
        return {"pwmfan": {"speed": [50], "rpm": [1200]}}

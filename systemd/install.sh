#!/usr/bin/env bash
# Installs mqtt_telemetry.py and mqtt_llm.py as systemd services on a Jetson Orin Nano.
# Run as root. Safe to re-run: skips anything already in place, never overwrites
# an existing MQTT credential file.
set -euo pipefail

usage() {
    echo "Usage: sudo $0 --broker <host> --username <mqtt-user>" >&2
    exit 1
}

BROKER=""
MQTT_USER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --broker) BROKER="$2"; shift 2 ;;
        --username) MQTT_USER="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[[ -n "$BROKER" && -n "$MQTT_USER" ]] || usage
[[ "$(id -u)" -eq 0 ]] || { echo "Run as root (sudo)." >&2; exit 1; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR=/opt/orin-mqtt
CONF_DIR=/etc/orin-mqtt
CRED_FILE="$CONF_DIR/mqtt_password"
UNIT_DIR=/etc/systemd/system

if ! getent passwd orin-mqtt >/dev/null; then
    useradd --system --no-create-home --shell /usr/sbin/nologin --comment "Orin MQTT services" orin-mqtt
fi

if getent group jtop >/dev/null; then
    usermod -a -G jtop orin-mqtt
else
    echo "warning: no 'jtop' group found - install jetson-stats first, mqtt_telemetry.py needs it to reach jtop.sock" >&2
fi

mkdir -p "$INSTALL_DIR"
install -m 755 -o orin-mqtt -g orin-mqtt "$REPO_DIR/mqtt_telemetry.py" "$INSTALL_DIR/mqtt_telemetry.py"
install -m 755 -o orin-mqtt -g orin-mqtt "$REPO_DIR/mqtt_llm.py" "$INSTALL_DIR/mqtt_llm.py"

mkdir -p "$CONF_DIR"
if [[ ! -f "$CRED_FILE" ]]; then
    install -m 600 -o root -g root /dev/null "$CRED_FILE"
    echo "Created $CRED_FILE - edit it now and put the MQTT password inside before starting the services."
else
    echo "$CRED_FILE already exists, leaving it untouched."
fi

for svc in orin-mqtt-telemetry orin-mqtt-llm; do
    sed -e "s/<HOST_IP>/$BROKER/g" -e "s/<USERNAME>/$MQTT_USER/g" \
        "$REPO_DIR/systemd/$svc.service" > "$UNIT_DIR/$svc.service"
    chmod 644 "$UNIT_DIR/$svc.service"
done

systemctl daemon-reload

cat <<EOF

Installed. Next steps:
  1. Confirm the MQTT password in $CRED_FILE.
  2. sudo systemctl enable --now orin-mqtt-telemetry.service orin-mqtt-llm.service
  3. sudo systemctl status orin-mqtt-telemetry.service orin-mqtt-llm.service
EOF

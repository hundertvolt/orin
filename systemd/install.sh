#!/usr/bin/env bash
# Installs and starts mqtt_telemetry.py and mqtt_llm.py as systemd services on a
# Jetson Orin Nano. Run as root. Safe to re-run: skips anything already in place,
# and re-applies the given password/broker/username each time so re-running with
# updated values doubles as a way to rotate them.
#
# Note: --password on the command line ends up in shell history and briefly in
# `ps` output for other users on the box. Fine for a single-user device behind a
# router; clear your shell history afterwards if that matters to you.
set -euo pipefail

usage() {
    echo "Usage: sudo $0 --broker <host> --username <mqtt-user> --password <mqtt-password>" >&2
    exit 1
}

BROKER=""
MQTT_USER=""
MQTT_PASSWORD=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --broker) BROKER="$2"; shift 2 ;;
        --username) MQTT_USER="$2"; shift 2 ;;
        --password) MQTT_PASSWORD="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[[ -n "$BROKER" && -n "$MQTT_USER" && -n "$MQTT_PASSWORD" ]] || usage
[[ "$(id -u)" -eq 0 ]] || { echo "Run as root (sudo)." >&2; exit 1; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR=/opt/orin-mqtt
CONF_DIR=/etc/orin-mqtt
CRED_FILE="$CONF_DIR/mqtt_password"
UNIT_DIR=/etc/systemd/system
SERVICES=(orin-mqtt-telemetry orin-mqtt-llm)

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
install -m 600 -o root -g root /dev/null "$CRED_FILE"
printf '%s' "$MQTT_PASSWORD" > "$CRED_FILE"

for svc in "${SERVICES[@]}"; do
    sed -e "s/<HOST_IP>/$BROKER/g" -e "s/<USERNAME>/$MQTT_USER/g" \
        "$REPO_DIR/systemd/$svc.service" > "$UNIT_DIR/$svc.service"
    chmod 644 "$UNIT_DIR/$svc.service"
done

SERVICE_UNITS=("${SERVICES[@]/%/.service}")
systemctl daemon-reload
systemctl enable "${SERVICE_UNITS[@]}"
systemctl start "${SERVICE_UNITS[@]}"

cat <<EOF

Installed, enabled, and started: ${SERVICE_UNITS[*]}

Check status:
  sudo systemctl status orin-mqtt-telemetry.service
  sudo systemctl status orin-mqtt-llm.service

Follow logs:
  sudo journalctl -u orin-mqtt-telemetry.service -f
  sudo journalctl -u orin-mqtt-llm.service -f
EOF

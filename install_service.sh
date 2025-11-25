#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SERVICE_NAME="hakan-bey-bot.service"
SYSTEMD_PATH="/etc/systemd/system/${SERVICE_NAME}"
BOT_USER="botuser"
WORKING_DIR="/home/${BOT_USER}/hakan-bey-bot"
PYTHON_BIN="${WORKING_DIR}/.venv/bin/python"

info() {
  printf '\n[INFO] %s\n' "$1"
}

ensure_logs_dir() {
  if [[ ! -d "${REPO_ROOT}/logs" ]]; then
    info "Creating logs directory"
    mkdir -p "${REPO_ROOT}/logs"
  fi
}

create_venv() {
  if [[ ! -d "${REPO_ROOT}/.venv" ]]; then
    info "Creating Python virtual environment at ${REPO_ROOT}/.venv"
    python3 -m venv "${REPO_ROOT}/.venv"
  else
    info "Virtual environment already exists; reusing ${REPO_ROOT}/.venv"
  fi
  info "Upgrading pip and installing requirements"
  "${REPO_ROOT}/.venv/bin/pip" install --upgrade pip
  "${REPO_ROOT}/.venv/bin/pip" install -r "${REPO_ROOT}/requirements.txt"
}

write_service_file() {
  info "Writing systemd unit to ${SYSTEMD_PATH}"
  cat <<EOF > "${SYSTEMD_PATH}"
[Unit]
Description=Hakan Bey Binance USDT Spot Signal Bot
After=network.target

[Service]
Type=simple
User=${BOT_USER}
WorkingDirectory=${WORKING_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=${PYTHON_BIN} -m src.main
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
}

reload_and_enable() {
  info "Reloading systemd daemon"
  systemctl daemon-reload
  info "Enabling ${SERVICE_NAME}"
  systemctl enable "${SERVICE_NAME}"
  info "Restarting ${SERVICE_NAME}"
  systemctl restart "${SERVICE_NAME}"
}

main() {
  info "Repository root detected at ${REPO_ROOT}"
  ensure_logs_dir
  create_venv
  write_service_file
  reload_and_enable
  printf '\nService installed as %s.\n' "${SERVICE_NAME}"
  printf 'Use:\n  sudo systemctl status %s\n  sudo journalctl -u %s -f\n' "${SERVICE_NAME}" "${SERVICE_NAME}"
}

main "$@"

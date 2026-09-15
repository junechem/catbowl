#!/usr/bin/env bash
# The Pi-side fixes from 2026-09-14: keep logs across reboots, Wi-Fi power
# saving off, and a monitor that logs reachability, power, temperature and
# memory every 15 s to ~/netmon.log. Safe to run again.
#
#   ./scripts/install_pi_health.sh
set -euo pipefail
cd "$(dirname "$0")/.."

sudo mkdir -p /etc/systemd/journald.conf.d /etc/NetworkManager/conf.d
sudo install -m 644 pi/journald-keep-logs.conf /etc/systemd/journald.conf.d/90-keep-logs.conf
sudo systemctl restart systemd-journald
sudo journalctl --flush

sudo install -m 644 pi/wifi-powersave-off.conf /etc/NetworkManager/conf.d/wifi-powersave-off.conf
# Takes effect at the next connection; this turns it off now without dropping it.
sudo /usr/sbin/iw dev wlan0 set power_save off 2>/dev/null || true

sudo install -m 755 pi/netmon.sh /usr/local/bin/netmon.sh
sudo install -m 644 systemd/netmon.service /etc/systemd/system/netmon.service
sudo systemctl daemon-reload
sudo systemctl enable --now netmon

echo "journal:    $(sudo ls /var/log/journal | head -1)"
echo "power save: $(sudo /usr/sbin/iw dev wlan0 get power_save 2>/dev/null || echo unknown)"
echo "netmon:     $(systemctl is-active netmon), logging to ~/netmon.log"

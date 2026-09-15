#!/bin/bash
# One line every 15 s: can the Pi reach its router, and what state is it in.
# Written to disk and synced, so the last lines survive a hard crash.
LOG=${NETMON_LOG:-/home/rjweldon/netmon.log}
GW=$(ip route | awk '/default/ {print $3; exit}')
while true; do
  gw=${GW:-$(ip route | awk '/default/ {print $3; exit}')}
  if ping -c1 -W2 "$gw" >/dev/null 2>&1; then net=ok; else net=FAIL; fi
  link=$(awk '/wlan0/ {printf "q=%s sig=%s", $3, $4}' /proc/net/wireless)
  assoc=$(cat /sys/class/net/wlan0/operstate 2>/dev/null)
  thr=$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)
  temp=$(vcgencmd measure_temp 2>/dev/null | cut -d= -f2)
  mem=$(awk '/MemAvailable/ {print int($2/1024)"M"}' /proc/meminfo)
  swap=$(awk '/SwapFree/ {f=$2} /SwapTotal/ {t=$2} END {print int((t-f)/1024)"M"}' /proc/meminfo)
  load=$(cut -d' ' -f1 /proc/loadavg)
  echo "$(date '+%F %T') gw=$gw net=$net wlan0=$assoc $link throttled=$thr temp=$temp avail=$mem swapused=$swap load=$load" >> "$LOG"
  sync "$LOG"
  sleep 15
done

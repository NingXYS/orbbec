#!/usr/bin/env bash
# 安装 Orbbec USB 权限。Astra Pro Plus: 深度 2bc5:060f，彩色 2bc5:050f。
set -euo pipefail
if [[ "$(id -u)" -ne 0 ]]; then
  echo "请用 sudo 运行: sudo bash scripts/install_udev.sh"
  exit 1
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/third_party/pyorbbecsdk-v1/99-obsensor-libusb.rules"
DST="/etc/udev/rules.d/99-obsensor-libusb.rules"
if [[ -f "${SRC}" ]]; then
  cp "${SRC}" "${DST}"
else
  echo "找不到官方 rules，写入通用 2bc5 规则"
fi
# 兜底：所有 Orbbec USB 设备对 plugdev/video 可读写
cat >/etc/udev/rules.d/99-orbbec-all.rules <<'EOF'
# Orbbec 全系列（含 Astra Pro Plus 0x050f / 0x060f）
SUBSYSTEM=="usb", ATTR{idVendor}=="2bc5", MODE="0666"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="2bc5", MODE="0666"
EOF
udevadm control --reload-rules
udevadm trigger
echo "udev 已安装。请拔插一次相机，或重新登录后再取流。"

#!/usr/bin/env bash
set -euo pipefail

setup_can() {
    local device="$1"
    local interface="$2"

    if ip link show "$interface" >/dev/null 2>&1; then
        if ip link show "$interface" | head -n 1 | grep -q "<.*UP.*>"; then
            echo "$interface already UP"
        else
            echo "$interface exists; bringing it UP"
            sudo ip link set "$interface" up
        fi
        return
    fi

    echo "Creating $interface from $device"
    sudo slcand -o -f -s8 "$device" "$interface"
    sudo ip link set "$interface" up
}

echo "Setting up arm CAN interfaces..."

setup_can /dev/arxcan1 can0
setup_can /dev/arxcan3 can1

echo "Setting up button CAN interface..."

setup_can /dev/arxcan6 can6

echo
echo "Current CAN/link status:"
ip -br link

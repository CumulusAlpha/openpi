# -- coding: UTF-8
import shutil
import subprocess


def ensure_can_interfaces(interfaces, root=None):
    """Bring CAN interfaces up when the local setup script is available."""
    interfaces = [interface for interface in interfaces if interface]
    if not interfaces:
        return

    ip_cmd = shutil.which("ip")
    if ip_cmd is None:
        print("Cannot check CAN interfaces: `ip` command not found.")
        return

    missing = []
    for interface in interfaces:
        result = subprocess.run(
            [ip_cmd, "link", "show", interface],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            missing.append(interface)

    if not missing:
        return

    setup_script = None
    if root is not None:
        candidate = getattr(root, "joinpath", None)
        setup_script = candidate("utils", "setup_can.sh") if candidate else None

    if setup_script is not None and setup_script.exists():
        print(f"CAN interfaces not found {missing}; trying {setup_script}")
        subprocess.run(["bash", str(setup_script)], check=False)
    else:
        print(f"CAN interfaces not found and no setup script is available: {missing}")

"""Read-only diagnostic: can this Python environment import TensorFlow, and if not, why?

Usage (from the backend folder, with your virtual environment active):
    python diagnose_tf.py

It changes nothing on your system. The import is tried in a separate process so that a crash or a blocked
DLL cannot take this script down.
"""
import platform
import subprocess
import sys

from predictor import classify_import_error, HINTS


def main() -> int:
    print(f"Python      : {sys.version.split()[0]} ({platform.system()} {platform.release()})")
    print(f"Interpreter : {sys.executable}")

    probe = (
        "import tensorflow as tf, numpy as np;"
        "print('tensorflow', tf.__version__, '| numpy', np.__version__)"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)

    if proc.returncode == 0:
        print("RESULT      : OK -", proc.stdout.strip())
        print("TensorFlow imports fine. If the API still reports 'not ready', check `GET /api/ready` for the reason.")
        return 0

    stderr = proc.stderr.strip()
    last_line = stderr.splitlines()[-1] if stderr else "(no error text; process exit code %d)" % proc.returncode
    exc = ModuleNotFoundError(last_line) if "ModuleNotFoundError" in last_line else ImportError(last_line)
    code, message = classify_import_error(exc)

    print("RESULT      : TensorFlow could NOT be imported")
    print(f"Diagnosis   : {code}")
    print(f"Meaning     : {message}")
    print(f"Python error: {last_line}")
    hint = HINTS.get(code)
    if hint:
        print(f"What to do  : {hint}")

    if code == "tensorflow_blocked_by_policy":
        print(
            "\nThis is Windows security, not a broken install - reinstalling TensorFlow will not help.\n"
            "To confirm and report it to your administrator, run in PowerShell:\n"
            '  Get-WinEvent -LogName "Microsoft-Windows-CodeIntegrity/Operational" -MaxEvents 50 |\n'
            "    Where-Object { $_.Id -in 3033,3077,3089 } | Format-List TimeCreated, Id, Message\n"
            "Give them the blocked file name (e.g. ...\\tensorflow\\python\\framework\\_dtypes.pyd) and the policy ID."
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

# !/usr/bin/python
# coding=utf-8
"""
Test script for ExecutionMonitor with Maya Connection.
Run this script to verify that the Execution Monitor dialog appears and functions correctly
when blocking Maya operations take too long.
"""
import sys
import logging
import time

# Ensure paths are set up if running from _scripts root
sys.path.append(r"o:\Cloud\Code\_scripts\mayatk")
sys.path.append(r"o:\Cloud\Code\_scripts\pythontk")

from mayatk.env_utils.maya_connection import MayaConnection
from pythontk.core_utils.execution_monitor import ExecutionMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("MonitorTest")

MONITOR_THRESHOLD = 3.0  # seconds
MAYA_SLEEP_TIME = 20.0  # seconds


@ExecutionMonitor.execution_monitor(
    threshold=MONITOR_THRESHOLD,
    message="Maya is sleeping (Simulation)",
    logger=logger,
    show_dialog=True,
    allow_escape_cancel=True,
)
def run_long_operation(conn):
    """
    Executes a long running sleep command in Maya.
    """
    logger.info(
        f"Starting blocking operation in Maya (Duration: {MAYA_SLEEP_TIME}s)..."
    )
    logger.info(f"Monitor should appear after {MONITOR_THRESHOLD}s.")

    # We use a simple sleep in Maya
    # Note: wait_for_response=True will make the client (this script) wait for Maya to finish.
    code = f"import time; print('Maya starting sleep...'); time.sleep({MAYA_SLEEP_TIME}); print('Maya finished sleep!')"

    # We use capture_output=True to verify we get the print output from Maya
    # We set a large timeout for the socket so it doesn't time out network-wise before the monitor kicks in
    result = conn.execute(code, timeout=MAYA_SLEEP_TIME + 5, capture_output=True)

    logger.info(f"Operation completed! Output from Maya:\n{result}")
    return result


def main():
    print("==================================================")
    print("   ExecutionMonitor + MayaConnection Test         ")
    print("==================================================")

    conn = MayaConnection.get_instance()

    # Connect with launch=True to automatically start Maya if needed
    logger.info("Connecting to Maya (will launch if not running)...")
    if not conn.connect(mode="auto", port=7002, launch=True):
        logger.error("Failed to connect to Maya.")
        return

    print(f"Connected to Maya in mode: {conn.mode}")

    try:
        print("\n--- TEST: Long Running Operation ---")
        run_long_operation(conn)

    except KeyboardInterrupt:
        print(
            "\n[!] Caught KeyboardInterrupt - Operation Cancelled by User (via Monitor)."
        )
    except Exception as e:
        print(f"\n[!] Caught Exception: {e}")
    finally:
        # Optional: Ask to close Maya if we launched it?
        # For now, we leave it open for inspection or subsequent tests.
        pass

    print("\nTest finished.")


if __name__ == "__main__":
    main()

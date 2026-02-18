# !/usr/bin/python
# coding=utf-8
import os
import sys
import time
import argparse
import subprocess

# Add pythontk to path for standalone execution
try:
    import pythontk
except ImportError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    import pythontk

from pythontk.core_utils.app_launcher import AppLauncher


def launch_substance_painter(file_path=None, headless=False, enable_remote=True):
    """
    Launch Adobe Substance 3D Painter using AppLauncher.
    
    Args:
        file_path (str): Path to a mesh (.obj/fbx) or project (.spp) file.
        headless (bool): If True, launch in automation mode (technically GUI still shows, but setup for remote control).
        enable_remote (bool): Enable the remote scripting API.
    """
    app_name = "Adobe Substance 3D Painter"
    
    # Arguments
    args = []
    
    if file_path:
        if file_path.lower().endswith(".spp"):
            # Project file is passed as direct argument
            args.append(file_path)
        else:
            # Mesh file requires --mesh flag
            args.extend(["--mesh", file_path])
    
    if enable_remote or headless:
        args.append("--enable-remote-scripting")
    
    # Substance Painter doesn't have a true --headless flag, but automation pipelines
    # often minimize the window or run on a render node.
    # We use 'detached' so this script doesn't block.
    
    cwd = os.path.dirname(file_path) if file_path else None
    
    print(f"Launching {app_name}...")
    print(f"Args: {args}")
    
    proc = AppLauncher.launch(
        app_name, 
        args=args, 
        cwd=cwd, 
        detached=True
    )
    
    if proc:
        print(f"Successfully launched Substance Painter (PID: {proc.pid})")
        if headless:
            print("Running in headless automation mode (Remote Scripting Enabled).")
            # In a real pipeline, you would now connect to localhost:3000 using JSON-RPC
            # to drive the application, export textures, and then call application.close()
    else:
        print("Failed to launch Substance Painter. Ensure it is installed and in your PATH.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Substance Painter via AppLauncher")
    parser.add_argument("--file", "-f", help="Path to mesh or project file")
    parser.add_argument("--headless", action="store_true", help="Launch in headless automation mode")
    parser.add_argument("--gui", action="store_true", help="Launch in standard GUI mode")
    
    args = parser.parse_args()
    
    # Default to GUI if not specified
    is_headless = args.headless
    
    launch_substance_painter(
        file_path=args.file,
        headless=is_headless,
        enable_remote=True
    )

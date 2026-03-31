from __future__ import annotations

import socket
import sys
from pathlib import Path

# Add project root to Python path so we can import 'app'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import uvicorn

from app.settings import ApiServerSettings


def get_available_port(host: str, start_port: int, max_tries: int = 25) -> int:
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find an available port starting from {start_port}.")


if __name__ == "__main__":
    reload_enabled = ApiServerSettings.API_RELOAD
    api_port = ApiServerSettings.API_PORT
    host = ApiServerSettings.API_HOST
    chosen_port = get_available_port(host=host, start_port=api_port)
    if chosen_port != api_port:
        print(f"Port {api_port} is busy. Starting API on {chosen_port}.")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=chosen_port,
        reload=reload_enabled,
        app_dir=str(PROJECT_ROOT),
    )

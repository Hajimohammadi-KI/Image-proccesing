from __future__ import annotations

import argparse
import functools
import os
import shutil
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the training progress dashboard.")
    parser.add_argument("--run-dir", required=True, help="Path to the run directory containing progress.json.")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory '{run_dir}' does not exist.")

    progress_file = run_dir / "progress.json"
    if not progress_file.exists():
        raise FileNotFoundError(
            f"{progress_file} not found. Start training first so progress.json is generated."
        )

    dashboard_src = Path(__file__).resolve().parents[1] / "web" / "progress_dashboard.html"
    dashboard_dst = run_dir / "progress_dashboard.html"
    shutil.copyfile(dashboard_src, dashboard_dst)

    handler = functools.partial(SimpleHTTPRequestHandler, directory=run_dir.as_posix())
    httpd = ThreadingHTTPServer(("", args.port), handler)
    url = f"http://localhost:{args.port}/progress_dashboard.html"
    print(f"Serving progress dashboard from {run_dir}")
    print(f"Open {url}")
    try:
        webbrowser.open(url, new=2)
    except Exception:
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()


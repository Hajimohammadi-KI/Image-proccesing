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
    parser.add_argument(
        "--dashboard-dir",
        default=None,
        help=(
            "Path to the built Next.js dashboard (defaults to web/progress-dashboard/out). "
            "Run `npm install` and `npm run build` inside web/progress-dashboard first."
        ),
    )
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

    if args.dashboard_dir:
        build_dir = Path(args.dashboard_dir).resolve()
    else:
        build_dir = Path(__file__).resolve().parents[1] / "web" / "progress-dashboard" / "out"

    if not build_dir.exists():
        raise FileNotFoundError(
            f"{build_dir} not found. Build the dashboard first using `npm run build` inside "
            "web/progress-dashboard/."
        )

    dashboard_dst = run_dir / "dashboard"
    if dashboard_dst.exists():
        shutil.rmtree(dashboard_dst)
    shutil.copytree(build_dir, dashboard_dst)

    handler = functools.partial(SimpleHTTPRequestHandler, directory=run_dir.as_posix())
    httpd = ThreadingHTTPServer(("", args.port), handler)
    url = f"http://localhost:{args.port}/dashboard/"
    print(f"Serving progress dashboard from {run_dir}")
    print(f"Dashboard assets copied from {build_dir}")
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


from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CatPred web API server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Install optional web dependencies with `pip install .[web]`.")
        return 1

    from .app import create_app

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

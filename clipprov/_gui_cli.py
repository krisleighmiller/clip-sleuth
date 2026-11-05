"""Checkout-pinned launcher for GUI-spawned CLI children.

The dev venv carries an editable install whose meta-path finder wins over
sys.path, so ``python -m clipprov.cli`` always imports the installed copy.
This script is launched by absolute path from the GUI controller; it registers
*this* checkout's package before any ``clipprov`` import, then hands off to
``clipprov.cli``. Keep in sync with the identical pin in the root conftest.py.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def _pin_checkout_package() -> None:
    spec = importlib.util.spec_from_file_location(
        "clipprov",
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load checkout package at {PACKAGE_ROOT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["clipprov"] = module
    spec.loader.exec_module(module)


def main() -> None:
    if not getattr(sys, "frozen", False):
        _pin_checkout_package()
    if len(sys.argv) == 3 and sys.argv[1] == "--gui-resolve-report":
        import json

        from clipprov.batch import current_report_dir

        try:
            report = current_report_dir(Path(sys.argv[2])) / "report.html"
            print(json.dumps({"uri": report.as_uri()}, sort_keys=True))
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        return
    from clipprov.cli import main as cli_main

    if getattr(sys, "frozen", False):
        if "LD_LIBRARY_PATH_ORIG" in os.environ:
            os.environ["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH_ORIG"]
        else:
            os.environ.pop("LD_LIBRARY_PATH", None)
    cli_main()


if __name__ == "__main__":
    main()

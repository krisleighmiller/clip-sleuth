"""Linux desktop bundle entry point, including its owned CLI worker."""

import sys

from clipprov import _gui_cli, gui

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--gui-worker":
        sys.argv.pop(1)
        _gui_cli.main()
    else:
        gui.main()

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || -e $1 ]]; then
    printf 'Usage: PYTHON=/path/to/venv/bin/python %s NEW_OUTPUT_DIR\n' "$0" >&2
    exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
out=$1
python=${PYTHON:-python3}
if [[ -n $(git -C "$root" status --porcelain) ]]; then
    printf 'Build from a clean public checkout, not uncommitted files.\n' >&2
    exit 2
fi
mkdir -p "$out"
out=$(cd -- "$out" && pwd)
version=$("$python" -c 'import pathlib, tomllib, sys; print(tomllib.loads((pathlib.Path(sys.argv[1]) / "pyproject.toml").read_text())["project"]["version"])' "$root")
PYINSTALLER_CONFIG_DIR="$out/cache" "$python" -m PyInstaller --noconfirm --clean \
    --distpath "$out/dist" --workpath "$out/build" "$root/packaging/clip-sleuth.spec"
mkdir -p "$out/archive"
cp -a "$out/dist/clip-sleuth" "$out/archive/"
cp "$root/LICENSE" "$out/archive/clip-sleuth/LICENSE"
cp "$root/packaging/README-linux.md" "$out/archive/clip-sleuth/README-LINUX.md"
archive="$out/clip-sleuth-$version-linux-x86_64.tar.gz"
tar -C "$out/archive" -czf "$archive" clip-sleuth
printf '%s\n' "$archive"

# -*- mode: python ; coding: utf-8 -*-
"""Linux onedir bundle; preserve source bytes used by provenance receipts."""

from pathlib import Path

root = Path(SPECPATH).resolve().parent
sources = sorted((root / "clipprov").glob("*.py"))
assert sources and (root / "clipprov" / "__init__.py") in sources

a = Analysis(
    [str(root / "packaging" / "clip_sleuth.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[(str(path), "clipprov") for path in sources] + [(str(root / "LICENSE"), ".")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="clip-sleuth", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="clip-sleuth")

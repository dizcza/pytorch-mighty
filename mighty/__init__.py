def _get_version():
    from pathlib import Path
    vpath = Path(__file__).parent / "VERSION"
    with open(vpath) as f:
        version = f.read()
    return version


__version__ = _get_version()

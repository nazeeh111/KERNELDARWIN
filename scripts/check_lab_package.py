"""Check that a built wheel contains the exact current-source workbench assets."""
from pathlib import Path
import sys
import zipfile


def main() -> None:
    source = Path(__file__).resolve().parents[1] / "lab"
    wheel = Path(sys.argv[1])
    count = 0
    with zipfile.ZipFile(wheel) as archive:
        for path in source.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(source).as_posix()
            if archive.read(f"kerneldarwin/lab/{relative}") != path.read_bytes():
                raise SystemExit(f"Packaged LAB asset differs: {relative}")
            count += 1
    print(f"Verified {count} packaged LAB files byte-for-byte against current source.")


if __name__ == "__main__":
    main()

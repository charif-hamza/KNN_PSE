"""Download the CSTH dataset splits from Zenodo with checksum verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

CHUNK_SIZE = 1 << 20
BASE_URL = "https://zenodo.org/records/10093059/files"
EXPECTED_SHA256: dict[str, str] = {
    "train.pt": "25cd483c3107ffac44d65df0d7bf5fd8e578778489a52d5649772898276300bb",
    "val.pt": "4f350ed2a08a2af30789cd8f303bf07098c14c55b9854660f889aa0ffc01c7da",
    "test.pt": "cb3a3334d2ad420460327c1dfba57d214017caee73244d1c52f8fea36f5c628e",
}


def download_file(filename: str, output_dir: Path) -> Path:
    """Download *filename* into *output_dir* while streaming the checksum."""

    target = output_dir / filename
    url = f"{BASE_URL}/{filename}?download=1"
    hasher = hashlib.sha256()
    with urlopen(url) as response, open(target, "wb") as handle:  # type: ignore[arg-type]
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            hasher.update(chunk)
    digest = hasher.hexdigest()
    expected = EXPECTED_SHA256[filename]
    if digest != expected:
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {filename}: expected {expected}, got {digest}"
        )
    return target


def write_manifest(output_dir: Path) -> None:
    """Persist the checksums for provenance tracking."""

    manifest = {name: sha for name, sha in EXPECTED_SHA256.items()}
    with open(output_dir / "checksums.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CSTH dataset splits")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if files exist"
    )
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in EXPECTED_SHA256:
        target = output_dir / filename
        if target.exists() and not args.force:
            print(f"Skipping {filename} (already present)")
            continue
        print(f"Downloading {filename}...")
        download_file(filename, output_dir)
        print(f"✔ Downloaded {filename}")

    write_manifest(output_dir)
    print(f"Checksums written to {output_dir / 'checksums.json'}")


if __name__ == "__main__":  # pragma: no cover
    main()

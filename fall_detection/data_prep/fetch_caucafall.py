"""Download the CAUCAFall dataset from Mendeley Data into data/raw/caucafall/.

CAUCAFall (Universidad del Cauca) - Eduardo Nino et al. - is published on
Mendeley Data. It is a single large archive (~3 GB); expect a long download.

    python -m fall_detection.data_prep.fetch_caucafall
    python -m fall_detection.data_prep.fetch_caucafall --dataset-id 7w7fcy7ky --version 4

Mendeley occasionally changes the dataset id / hosting. If this fails, open the
dataset page in a browser, download the zip, and extract it to
``fall_detection/data/raw/caucafall/`` yourself - the adapter only needs the
per-clip folders and their annotation CSVs.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

DEFAULT_ID = "7w7fcy7ky"
DEFAULT_VERSION = 4
API = "https://data.mendeley.com/public-api/datasets"


def _list_files(dataset_id: str, version: int) -> list[dict]:
    import requests

    url = f"{API}/{dataset_id}/files?version={version}"
    r = requests.get(url, timeout=60, headers={"Accept": "application/json"})
    r.raise_for_status()
    return r.json()


def _download(url: str, dest: Path, *, retries: int = 5, timeout: float = 300.0) -> bool:
    import requests

    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            tmp.replace(dest)
            print(f"  ok  {dest.name} ({dest.stat().st_size:,} B)")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt}/{retries} {dest.name}: {e}")
            tmp.unlink(missing_ok=True)
    return False


def fetch(root: Path, dataset_id: str, version: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    files = _list_files(dataset_id, version)
    print(f"{len(files)} file(s) in Mendeley dataset {dataset_id} v{version}")

    for f in files:
        name = f.get("filename") or f.get("name") or "download.bin"
        url = (f.get("content_details") or {}).get("download_url") or f.get("download_url")
        if not url:
            print(f"  SKIP {name}: no download url in API response")
            continue
        dest = root / name
        if not _download(url, dest):
            print(f"  FAILED {name}")
            continue
        if dest.suffix.lower() == ".zip":
            print(f"  extracting {dest.name} ...")
            with zipfile.ZipFile(dest) as z:
                z.extractall(root)


def _main(argv: list[str] | None = None) -> None:
    from ..common.paths import load_config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset-id", default=DEFAULT_ID)
    ap.add_argument("--version", type=int, default=DEFAULT_VERSION)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    root = Path(cfg.path("data_raw")) / "caucafall"
    print(f"downloading CAUCAFall -> {root}")
    try:
        fetch(root, args.dataset_id, args.version)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"Mendeley download failed ({e}). Download the archive manually from "
            f"https://data.mendeley.com/datasets/{args.dataset_id} and extract to {root}"
        )


if __name__ == "__main__":
    _main()

"""Download the UR Fall Detection Dataset (camera 0) into data/raw/urfd/.

Grabs the two annotation CSVs and the cam0 RGB videos for every fall + ADL
sequence, with retry/resume. Run:

    python -m fall_detection.data_prep.fetch_urfd            # everything (cam0 mp4)
    python -m fall_detection.data_prep.fetch_urfd --limit 4  # first 4 of each (quick)
    python -m fall_detection.data_prep.fetch_urfd --rgb-zip  # PNG sequences instead of mp4

The dataset is hosted by Michal Kepski, University of Rzeszow:
http://fenix.ur.edu.pl/~mkepski/ds/uf.html
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

BASE = "http://fenix.ur.edu.pl/~mkepski/ds/data"
N_FALLS = 30
N_ADLS = 40


def _download(url: str, dest: Path, *, retries: int = 5, timeout: float = 120.0) -> bool:
    import requests

    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    print(f"  404 {url}")
                    return False
                r.raise_for_status()
                expected = int(r.headers.get("Content-Length", 0))
                written = 0
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 16):
                        fh.write(chunk)
                        written += len(chunk)
            if expected and written != expected:
                raise OSError(f"short read {written}/{expected}")
            tmp.replace(dest)
            print(f"  ok  {dest.name}  ({written:,} B)")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt}/{retries} {dest.name}: {e}")
            tmp.unlink(missing_ok=True)
            time.sleep(2 * attempt)
    print(f"  FAILED {url}")
    return False


def fetch(root: Path, *, limit: int | None = None, rgb_zip: bool = False) -> dict:
    ann = root / "annotations"
    results = {"annotations": 0, "falls": 0, "adls": 0, "failed": []}

    for name in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        if _download(f"{BASE}/{name}", ann / name):
            results["annotations"] += 1
        else:
            results["failed"].append(name)

    def seq_targets(prefix: str, count: int, subdir: str):
        n = count if limit is None else min(limit, count)
        for i in range(1, n + 1):
            sid = f"{prefix}-{i:02d}"
            if rgb_zip:
                yield f"{BASE}/{sid}-cam0-rgb.zip", root / subdir / f"{sid}-cam0-rgb.zip"
            else:
                yield f"{BASE}/{sid}-cam0.mp4", root / subdir / f"{sid}-cam0.mp4"

    for url, dest in seq_targets("fall", N_FALLS, "falls"):
        if _download(url, dest):
            results["falls"] += 1
        else:
            results["failed"].append(dest.name)

    for url, dest in seq_targets("adl", N_ADLS, "adl"):
        if _download(url, dest):
            results["adls"] += 1
        else:
            results["failed"].append(dest.name)

    return results


def _main(argv: list[str] | None = None) -> None:
    from ..common.paths import load_config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N fall + N adl sequences")
    ap.add_argument("--rgb-zip", action="store_true", help="download PNG sequence zips instead of mp4")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    root = Path(cfg.path("data_raw")) / "urfd"
    print(f"downloading URFD (cam0) -> {root}")
    res = fetch(root, limit=args.limit, rgb_zip=args.rgb_zip)
    print(f"\nannotations: {res['annotations']}/2   falls: {res['falls']}   adls: {res['adls']}")
    if res["failed"]:
        print(f"failed ({len(res['failed'])}): {res['failed'][:10]}{' ...' if len(res['failed']) > 10 else ''}")


if __name__ == "__main__":
    _main()

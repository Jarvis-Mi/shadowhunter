#!/usr/bin/env python
"""Pointers to the three free datasets, and a checker for what you already have.

Nothing here downloads tens of gigabytes behind your back. Satellite archives
want an account or an AWS CLI call, so this script prints the exact command
for each source and then verifies whatever landed in ``data/raw``.

    python scripts/download_data.py --list
    python scripts/download_data.py --verify
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shadowhunter.core.config import SETTINGS  # noqa: E402

SOURCES = {
    "spacenet": {
        "what": "SpaceNet 2/6 — high-resolution imagery with building footprints "
                "and, in the Urban 3D / SN7 subsets, height labels. Best source "
                "for training the CNN regressor.",
        "licence": "CC BY-SA 4.0 — free, attribution required",
        "how": [
            "pip install awscli",
            "aws s3 ls s3://spacenet-dataset/ --no-sign-request",
            "aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/ "
            "data/raw/ --recursive --no-sign-request",
        ],
        "url": "https://spacenet.ai/datasets/",
    },
    "sentinel2": {
        "what": "Sentinel-2 L2A — 10 m multispectral, global, revisit every 5 days. "
                "Free forever. This is the archive Idea 3 (temporal selection) "
                "searches over for the longest, cleanest shadow.",
        "licence": "Copernicus open licence — free, no registration for the STAC API",
        "how": [
            "pip install pystac-client odc-stac",
            "# then query by bbox + date range against:",
            "#   https://earth-search.aws.element84.com/v1",
            "# filter on eo:cloud_cover < 10 and read the sun angles from the metadata",
        ],
        "url": "https://dataspace.copernicus.eu/",
    },
    "inria": {
        "what": "Inria Aerial Image Labeling — 0.3 m aerial tiles over 10 cities "
                "with building masks. No heights, but excellent for validating "
                "the shadow segmentation on real imagery.",
        "licence": "free for research, registration required",
        "how": ["# register, then download the .7z parts into data/raw/"],
        "url": "https://project.inria.fr/aerialimagelabeling/",
    },
}


def cmd_list() -> int:
    print("\nFREE DATASETS — nothing in this project needs a paid image.\n")
    for key, meta in SOURCES.items():
        print(f"  {key.upper()}")
        print(f"    {meta['what']}")
        print(f"    licence : {meta['licence']}")
        print(f"    url     : {meta['url']}")
        for line in meta["how"]:
            print(f"      $ {line}" if not line.startswith("#") else f"        {line}")
        print()
    print("  Or skip downloads entirely — the synthetic city generator is\n"
          "  physically consistent and runs the whole pipeline:\n")
    print("      $ python run.py synth --count 8\n")
    return 0


def cmd_verify() -> int:
    raw = Path(SETTINGS.data_dir) / "raw"
    synth = Path(SETTINGS.data_dir) / "synthetic"
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    real = [p for p in raw.glob("**/*") if p.suffix.lower() in exts]
    fake = [p for p in synth.glob("*") if p.suffix.lower() in exts]

    print(f"\n  data/raw       : {len(real)} scene(s), "
          f"{sum(p.stat().st_size for p in real) / 1e6:.1f} MB")
    print(f"  data/synthetic : {len(fake)} scene(s)")

    try:
        import rasterio  # noqa: F401

        print("  rasterio       : available (GeoTIFF supported)")
    except ImportError:
        print("  rasterio       : MISSING — GeoTIFFs will fall back to OpenCV, "
              "losing CRS and sun-angle metadata")

    for path in real[:5]:
        try:
            from shadowhunter.models.vision.preprocess import load_scene

            scene = load_scene(path)
            print(f"    ok  {path.name}  {scene.size[0]}x{scene.size[1]}  "
                  f"sun {scene.sun.elevation_deg:.0f}° @ {scene.sun.azimuth_deg:.0f}°  "
                  f"gsd {scene.sun.gsd_m:.2f} m")
        except Exception as exc:
            print(f"    !!  {path.name}: {type(exc).__name__}: {exc}")

    if not real and not fake:
        print("\n  no scenes yet — run:  python run.py synth --count 8\n")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="print dataset sources and commands")
    p.add_argument("--verify", action="store_true", help="check what is already on disk")
    args = p.parse_args()
    if args.verify:
        return cmd_verify()
    return cmd_list()


if __name__ == "__main__":
    raise SystemExit(main())

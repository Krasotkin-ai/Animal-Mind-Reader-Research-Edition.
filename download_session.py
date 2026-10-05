import argparse
from pathlib import Path
from dandi.dandiapi import DandiAPIClient

ap = argparse.ArgumentParser()
ap.add_argument("--dandiset", default="000231")
ap.add_argument("--asset", required=True)
ap.add_argument("--output", required=True)
args = ap.parse_args()

out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)

with DandiAPIClient() as client:
    ds = client.get_dandiset(args.dandiset, "draft")
    asset = ds.get_asset_by_path(args.asset)
    if asset is None:
        raise SystemExit(f"Asset not found: {args.asset}")
    print("Downloading:", asset.path)
    asset.download(path=out)
    print("Saved:", out)

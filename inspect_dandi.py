import argparse
from dandi.dandiapi import DandiAPIClient

ap = argparse.ArgumentParser()
ap.add_argument("--dandiset", default="000231")
args = ap.parse_args()

with DandiAPIClient() as client:
    ds = client.get_dandiset(args.dandiset, "draft")
    print("Dandiset:", ds.identifier)
    print("Name:", ds.name)
    print("\nFirst 100 assets:")
    for i, asset in enumerate(ds.get_assets()):
        print(asset.path)
        if i >= 99:
            break

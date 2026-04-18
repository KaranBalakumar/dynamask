from importlib import import_module
from tqdm import tqdm
from pathlib import Path
import os

try:
    _minio_mod = import_module("minio")
    _minio_err_mod = import_module("minio.error")
except ModuleNotFoundError:
    _minio_mod = None
    _minio_err_mod = None

Minio = getattr(_minio_mod, "Minio", None)
S3Error = getattr(_minio_err_mod, "S3Error", Exception)

# Do not change these prefilled constant
const_endpoint  = "airlab-share-01.andrew.cmu.edu:9000"
const_bucket    = "macvo-zed-data-icra25"
# End

class TqdmProgress:
    def __init__(self):
        self.pbar = None

    def set_meta(self, object_name: str, total_length: int):
        self.pbar = tqdm(total=total_length,unit="B",unit_scale=True,desc=f"Downloading {object_name}")

    def update(self, length: int):
        if self.pbar: self.pbar.update(length)

def main(args):
    if Minio is None:
        raise ImportError("minio is required to download ICRA25 Zed data.")

    download_dst = args.dst
    if not download_dst.exists():
        if input(f"{download_dst} does not exists. Want to create this folder? [y/n]").lower().strip() != 'y':
            raise Exception("Aborted by the user.")        
        download_dst.mkdir(parents=True, exist_ok=False)
    download_root = download_dst.resolve()

    if not args.access_key or not args.secret_key:
        raise ValueError(
            "Missing MinIO credentials. Set ICRA25_MINIO_ACCESS_KEY/ICRA25_MINIO_SECRET_KEY "
            "or pass --access-key/--secret-key."
        )

    client  = Minio(
        endpoint=const_endpoint, access_key=args.access_key, secret_key=args.secret_key, secure=True
    )
    
    objects = client.list_objects(const_bucket, recursive=True)
    for obj in objects:
        object_name = obj.object_name
        if object_name is None or obj.is_dir: continue
        
        dst = (download_root / object_name).resolve()
        if dst != download_root and download_root not in dst.parents:
            raise ValueError(f"Refusing to write outside destination root: {object_name}")
        
        if not dst.parent.exists(): dst.parent.mkdir(parents=True)
        
        try:
            client.fget_object(const_bucket, object_name, str(dst), progress=TqdmProgress())
        except S3Error as e:
            print(f"Error downloading {obj.object_name}: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog="MAC-VO ICRA 2025 Conference-day Zed Dataset Downloader")
    parser.add_argument("--dst", type=Path, help="Destination folder for")
    parser.add_argument("--access-key", type=str, default=os.getenv("ICRA25_MINIO_ACCESS_KEY", ""))
    parser.add_argument("--secret-key", type=str, default=os.getenv("ICRA25_MINIO_SECRET_KEY", ""))
    args   = parser.parse_args()

    main(args)

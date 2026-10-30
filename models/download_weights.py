"""
Model Weights Downloader for ESRGAN (x4)
Downloads official pretrained weights from reliable mirrors.
"""

import sys
import time
from pathlib import Path
import urllib.request
import urllib.error

# Add parent directory to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config


def download_file_with_progress(url: str, destination: Path) -> bool:
    """Download a file with console progress indication."""
    print(f"Connecting to: {url}")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SR-Vision-Lab/1.0 (Educational CVIP Project)"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024 * 64  # 64 KB
            downloaded = 0
            start_time = time.time()

            destination.parent.mkdir(parents=True, exist_ok=True)
            temp_path = destination.with_suffix(".tmp")

            with open(temp_path, "wb") as out_file:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out_file.write(buffer)
                    downloaded += len(buffer)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_done = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        elapsed = max(0.1, time.time() - start_time)
                        speed = mb_done / elapsed
                        sys.stdout.write(f"\rProgress: {percent:.1f}% ({mb_done:.1f}/{total_mb:.1f} MB) @ {speed:.2f} MB/s")
                    else:
                        sys.stdout.write(f"\rDownloaded: {downloaded / (1024 * 1024):.1f} MB")
                    sys.stdout.flush()

            sys.stdout.write("\n")
            if temp_path.exists():
                temp_path.replace(destination)
            print(f"Successfully downloaded to: {destination} ({destination.stat().st_size / (1024 * 1024):.1f} MB)")
            return True

    except Exception as e:
        print(f"\nDownload failed from {url}: {e}")
        temp_path = destination.with_suffix(".tmp")
        if temp_path.exists():
            temp_path.unlink()
        return False


def ensure_weights(destination: Path = None, verbose: bool = True) -> bool:
    """
    Ensure ESRGAN weights exist on disk.
    If missing, iterates through mirrors until successfully downloaded.
    """
    dest = destination or config.MODEL_WEIGHTS_PATH
    if dest.exists() and dest.stat().st_size > 10 * 1024 * 1024:
        if verbose:
            print(f"Pretrained weights already present at: {dest} ({dest.stat().st_size / (1024 * 1024):.1f} MB)")
        return True

    print(f"Pretrained weights not found at {dest}. Commencing automated download...")
    for mirror_url in config.MODEL_URLS:
        print(f"Attempting download from mirror: {mirror_url}")
        if download_file_with_progress(mirror_url, dest):
            # Verify file size (ESRGAN RRDB x4 weights are ~64-67 MB)
            if dest.stat().st_size > 50 * 1024 * 1024:
                return True
            else:
                print(f"Downloaded file appears incomplete ({dest.stat().st_size} bytes). Retrying...")
                dest.unlink(missing_ok=True)

    print("Error: Could not automatically download model weights from any mirror.")
    print("Please manually download 'RRDB_ESRGAN_x4.pth' and place it in the 'models/' folder.")
    return False


if __name__ == "__main__":
    success = ensure_weights()
    sys.exit(0 if success else 1)

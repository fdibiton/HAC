import argparse
import tarfile
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def count_from_tar(tar_path):
    parent_count = 0
    child_count = 0
    total_count = 0
    try:
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".jpg"):
                    if "parent" in member.name:
                        parent_count += 1
                    elif "child" in member.name:
                        child_count += 1
                    total_count += 1
    except tarfile.TarError as e:
        print(f"Failed to read {tar_path}: {e}")
    return parent_count, child_count, total_count

def count_images_in_tars(folder_path, num_workers):
    folder = Path(folder_path)
    tar_paths = sorted(folder.glob("*.tar"))

    total_parent = 0
    total_child = 0
    total_image = 0

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(count_from_tar, tar_paths), total=len(tar_paths)))

    for parent_count, child_count, total_count in results:
        total_parent += parent_count
        total_child += child_count
        total_image += total_count

    print(f"Parent images: {total_parent}")
    print(f"Child images: {total_child}")
    print(f"Total images: {total_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count parent and child images in TAR files.")
    parser.add_argument("-i", "--folder", type=str, required=True, help="Path to the folder containing TAR files.")
    parser.add_argument("-n", "--num-workers", type=int, default=cpu_count(), help="Number of parallel workers.")
    args = parser.parse_args()
    count_images_in_tars(args.folder, args.num_workers)

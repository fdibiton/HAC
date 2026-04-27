import argparse
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
import json
import sys


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # vqa root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
    
from vmcbench import write_image


def main():
    parser = argparse.ArgumentParser(description="Read and display Parquet file contents.")
    parser.add_argument("-i", "--parquet_file", type=Path, nargs='+', help="Path to the Parquet file")
    parser.add_argument("-o", "--output_dir", type=Path, help="Directory to save extracted images", required=True)
    args = parser.parse_args()
    
    # make output directory if it doesn't exist
    os.makedirs(args.output_dir / "images", exist_ok=True)
    
    # Read Parquet file(s)
    # if more than one file is given, concatenate them
    assert len(args.parquet_file) == 2, "Please provide exactly two Parquet files."
    print(f"Reading multiple Parquet files: {args.parquet_file} ...")
    df_list = [pd.read_parquet(pf) for pf in args.parquet_file]
    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {len(args.parquet_file)} files")
    
    num_zeros = len(str(len(df))) + 1
    ann_list = []
    num_skipped = 0
    
    # columns are: question, options, answer, image (base64-encoded JPEG)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        
        if row['image'] is None:
            #print(f"Skipping sample {idx} with no image.")
            num_skipped += 1
            continue
        
        # write image to file
        image_path = args.output_dir / "images" / row['image']['path'].rsplit('.', 1)[0]
        out_image_path = write_image(row['image']['bytes'], image_path)
        
        question = row['question'].strip()
        options = row['options']
        assert len(options) == 4, f"Expected 4 options for sample {idx}, got {len(options)}"
        
        right_answer_idx = int(row['answer'].strip())
        answer = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}[right_answer_idx]
        
        ann_list.append({
            "index": idx,
            "question": question,
            "options": {
                "A": options[0],
                "B": options[1],
                "C": options[2],
                "D": options[3],
            },
            "answer": answer,
            "image_path": str(os.path.basename(out_image_path))
        })
        
    print(f"Extracted images to {args.output_dir}")
    # serialize dictionary to ann.json in output directory
    ann_file = args.output_dir / "AI2D_ann.json"
    with open(ann_file, 'w') as f:
        json.dump(ann_list, f, indent=4)
    print("Annotation JSON files created.")
    
    print(f"Skipped {num_skipped} samples.")

        
if __name__ == "__main__":
    main()
        
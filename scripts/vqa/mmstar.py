#!/usr/bin/env python3
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
    parser.add_argument("-i", "--parquet_file", type=Path, help="Path to the Parquet file", required=True)
    parser.add_argument("-o", "--output_dir", type=Path, help="Directory to save extracted images", required=True)
    args = parser.parse_args()
    
    # make output directory if it doesn't exist
    os.makedirs(args.output_dir / "images", exist_ok=True)

    # Read Parquet file(s)
    print(f"Reading single Parquet file: {args.parquet_file} ...")
    df = pd.read_parquet(args.parquet_file)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    num_zeros = len(str(len(df))) + 1
    ann_list = []

    # columns are: index, question (question+options), image, answer, category, l2_category, meta-info dict
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        
        image_idx = row['meta_info']['image_path'].split('images/')[-1].split('.')[0]
        assert str(idx) == image_idx, f"Index mismatch: {idx} vs {image_idx}"
        
        if idx == 855:
            continue # question with 3 answers only
        
        # zero-pad index
        filled_idx = str(idx).zfill(num_zeros)
        # write image to file
        image_filename = f"{filled_idx}"
        image_path = args.output_dir / "images" / image_filename
        out_image_path = write_image(row['image'], image_path)
        
        try:
            if 'Options:' in row['question']:
                question, options = row['question'].split('Options: ')
                A = options.split('B: ')[0].split('A: ')[1]
                B = options.split('C: ')[0].split('B: ')[1]
                C = options.split('D: ')[0].split('C: ')[1]
                D = options.split('D: ')[1]
            elif 'Choices:' in row['question']:       
                row_question = row['question'].replace('Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: ', '')
                question, options = row_question.split('\nChoices:\n')
                A = options.split('(B)')[0].split('(A)')[1]
                B = options.split('(C)')[0].split('(B)')[1]
                C = options.split('(D)')[0].split('(C)')[1]
                D = options.split('(D)')[1]
        except Exception as e:
            print(f"Error parsing question/options for index {idx}: {row['question']}")
            import pdb; pdb.set_trace()
                
        # write question, options and answer to ann_dict
        ann_list.append({
            "index": idx,
            "question": question.strip(),
            "options": {
                "A": A.strip(),
                "B": B.strip(),
                "C": C.strip(),
                "D": D.strip()
            },
            "answer": row['answer'],
            "image_path": str(os.path.basename(out_image_path)),
            "category": row['category'],
            "l2_category": row['l2_category'],
        })
        
    print(f"Extracted images to {args.output_dir}")
    # serialize dictionary to ann.json in output directory
    ann_file = args.output_dir / "mmstar_ann.json"
    with open(ann_file, 'w') as f:
        json.dump(ann_list, f, indent=4)
    print("Annotation JSON files created.")

        
if __name__ == "__main__":
    main()

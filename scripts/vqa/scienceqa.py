import argparse
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
import json
import sys
import numpy as np

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
    num_skipped = 0
    num_5_answers = 0
    
    # columns are: image, question, choices, answer, hint, task, grade, subject, topic, category, skill, lecture, solution
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        
        if row['image'] is None:
            #print(f"Skipping sample {idx} with no image.")
            num_skipped += 1
            continue
        
        # zero-pad index
        filled_idx = str(idx).zfill(num_zeros)
        # write image to file
        image_filename = f"{filled_idx}"
        image_path = args.output_dir / "images" / image_filename
        out_image_path = write_image(row['image']['bytes'], image_path)
        
        question = row['question'].strip()
        options = row['choices'].tolist()

        right_answer_idx = row['answer']
        
        assert len(options) >= 2, f"Not enough options for sample {idx}: {options}"
            
        if len(options) < 4:
            # fill missing options with a wrong answer to make 4 options
            # take first wrong answer and duplicate it
            if right_answer_idx == 0:
                wrong_answer = options[-1]
            else:
                wrong_answer = options[0]
            while len(options) < 4:
                options.append(wrong_answer)
            #print(f"Sample {idx} had less than 4 options. Filled missing options with a wrong answer.")
            
        # turn answer into list A, B, C, D
        try:
            answer = ["A", "B", "C", "D"][right_answer_idx]
        except IndexError:
            assert right_answer_idx == 4, f"Unexpected answer index {right_answer_idx} for sample {idx}"
            num_5_answers += 1
            # overwrite answer with random choice among A, B, C, D
            idx_to_overwrite = np.random.randint(0, 4)
            options[idx_to_overwrite] = options[4]  # replace missing option with the 5th option
            answer = ["A", "B", "C", "D"][idx_to_overwrite]

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
    ann_file = args.output_dir / "ScienceQA_ann.json"
    with open(ann_file, 'w') as f:
        json.dump(ann_list, f, indent=4)
    print("Annotation JSON files created.")
    
    print(f"Skipped {num_skipped} samples. Samples with answer index 5 handled: {num_5_answers}.")

        
if __name__ == "__main__":
    # set numpy random seed for reproducibility
    np.random.seed(42)
    main()
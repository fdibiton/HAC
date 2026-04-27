import argparse
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
import json
import re
from io import BytesIO
from typing import Union, Mapping, Any
from PIL import Image


def save_hf_image_record(
    image_cell: Union[Mapping[str, Any], bytes, bytearray, str],
    out_path: Union[str, Path],
):
    image = Image.open(BytesIO(image_cell)).convert('RGB')
    image.save(out_path, format='WEBP')

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
    
    # columns are: question, answer, image (base64-encoded JPEG)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        
        # zero-pad index
        filled_idx = str(idx).zfill(num_zeros)
        # write image to file
        image_filename = f"{filled_idx}"
        image_path = args.output_dir / "images" / image_filename
        image_data = row['image']['bytes']
        out_image_path = os.path.join(args.output_dir, "images", image_filename + ".webp")
        save_hf_image_record(
            image_data,
            out_image_path,
        )
        # retain only multiple-choice answers (A, B, C, D)
        if not ('Please answer directly with only the letter of the correct option and nothing else.' in row['question']):
            #print(f"Skipping sample {idx} with non-multiple-choice question.")
            num_skipped += 1
            continue
        
        answer = row['answer'].strip()

        question, options = row['question'].split('\nA')
        question = question.strip()
        options = '\nA' + options.replace('Please answer directly with only the letter of the correct option and nothing else.', '')
        # replace any of A. B. C. with A B C
        options = options.replace('\nA. ', '\nA ').replace('\nB. ', '\nB ').replace('\nC. ', '\nC ').replace('\nD. ', '\nD ')
        # replace any of A: B: C: with A B C
        options = options.replace('\nA: ', '\nA ').replace('\nB: ', '\nB ').replace('\nC: ', '\nC ').replace('\nD: ', '\nD ')
        
        # each option is on a new line
        # \nA <option A>\nB <option B>\nC <option C>\nD <option D>\n
        options = re.findall(
            r'\n([A-D]) (.*?)(?=\n[A-D] |$)',
            options,
            re.DOTALL
        )
        options = [opt[1].strip() for opt in options]
        
        right_answer_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}.get(answer, None)
        if right_answer_idx is None:
            # look if answer is one of the options
            for opt_idx, opt_text in enumerate(options):
                if answer.strip() == opt_text:
                    right_answer_idx = opt_idx
                    answer = ['A', 'B', 'C', 'D'][opt_idx]
                    break
                raise ValueError(f"Answer '{answer}' not in ['A', 'B', 'C', 'D'] or matching options for sample {idx}.")
        
        if len(options) < 4:
            # fill missing options with a wrong answer to make 4 options
            # take first wrong answer and duplicate it
            if right_answer_idx == 0:
                wrong_answer = options[-1]
            else:
                wrong_answer = options[0]
            while len(options) < 4:
                options.append(wrong_answer)
                
        A = options[0].strip()
        B = options[1].strip()
        C = options[2].strip()
        D = options[3].strip()
        
        ann_list.append({
            "index": idx,
            "question": question,
            "options": {
                "A": A,
                "B": B,
                "C": C,
                "D": D
            },
            "answer": answer,
            "image_path": str(os.path.basename(out_image_path))
        })
        
    print(f"Extracted images to {args.output_dir}")
    # serialize dictionary to ann.json in output directory
    ann_file = args.output_dir / "RealWorldQA_ann.json"
    with open(ann_file, 'w') as f:
        json.dump(ann_list, f, indent=4)
    print("Annotation JSON files created.")
    
    print(f"Skipped {num_skipped} samples.")

        
if __name__ == "__main__":
    main()
        
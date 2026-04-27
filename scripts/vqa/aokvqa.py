import os
import sys
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import shutil

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # vqa root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
    
"""
JSON entry example:
{
    "split": "val",
    "image_id": 461751,
    "question_id": "22jbM6gDxdaMaunuzgrsBB",
    "question": "What is in the motorcyclist's mouth?",
    "choices": [
        "toothpick",
        "food",
        "popsicle stick",
        "cigarette"
    ],
    "correct_choice_idx": 3,
    "direct_answers": [
        ...
    ],
    "difficult_direct_answer": false,
    "rationales": [
        "He's smoking while riding.",
        "The motorcyclist has a lit cigarette in his mouth while he rides on the street.",
        "The man is smoking."
    ]
},
"""
    
def main():
    parser = argparse.ArgumentParser(description="Read and display JSON file contents.")
    parser.add_argument("-i1", "--json_file", type=Path, help="Path to the json file", required=True)
    parser.add_argument("-i2", "--input_dir", type=Path, help="Directory containing images", required=True)
    parser.add_argument("-o", "--output_dir", type=Path, help="Directory to save extracted images and annotations", required=True)
    args = parser.parse_args()
    
    # make output directory if it doesn't exist
    os.makedirs(args.output_dir / "images", exist_ok=True)
    
    # open and read json file
    with open(args.json_file, 'r') as f:
        data = json.load(f)
        
    ann_list = []
        
    # iterate through entries and save images and annotations
    for entry in tqdm(data, desc="Processing entries"):
        
        # 139 -> 000000000139 (filled with leading zeros)
        image_name = str(entry["image_id"]).zfill(12) + ".jpg"
        inp_image_path = args.input_dir / image_name
        out_image_path = args.output_dir / "images" / image_name
        
        # write question, options and answer to ann_dict
        ann_list.append({
            "index": entry["image_id"],
            "question": entry["question"],
            "options": {
                "A": entry["choices"][0],
                "B": entry["choices"][1],
                "C": entry["choices"][2],
                "D": entry["choices"][3]
            },
            "answer": {0: "A", 1: "B", 2: "C", 3: "D"}[entry["correct_choice_idx"]],
            "image_path": str(os.path.basename(out_image_path))
        })
        
        shutil.copy(inp_image_path, out_image_path)
        
    print(f"Extracted images to {args.output_dir}")
    # serialize dictionary to ann.json in output directory
    ann_file = args.output_dir / "aokvqa_ann.json"
    with open(ann_file, 'w') as f:
        json.dump(ann_list, f, indent=4)
    print("Annotation JSON files created.")
    
    
if __name__ == "__main__":
    main()
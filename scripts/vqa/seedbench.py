import argparse
import shutil
from pathlib import Path
import os
from tqdm import tqdm
import json


def main():
    parser = argparse.ArgumentParser(description="Read and display Parquet file contents.")
    parser.add_argument("-i", "--json_file", type=Path, help="Path to the JSON file", required=True)
    parser.add_argument("-o", "--output_dir", type=Path, help="Directory to save extracted images", required=True)
    args = parser.parse_args()
    
    # make output directory if it doesn't exist
    os.makedirs(args.output_dir / "images", exist_ok=True)
    
    # open and read json file
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    data = data['questions']
        
    ann_list = []
        
    # iterate through entries and save images and annotations
    for idx, entry in tqdm(enumerate(data), desc="Processing entries", total=len(data)):
        
        data_type = entry["data_type"]  # "image" or "video"
        if data_type != "image":
            continue  # skip videos
        
        image_name = entry["data_id"] + ".jpg"
        inp_image_path = os.path.dirname(args.json_file) + "/SEED-Bench-image/" + entry["data_id"]
        out_image_path = args.output_dir / "images" / image_name
        
        # write question, options and answer to ann_dict
        ann_list.append({
            "index": idx,
            "question": entry["question"],
            "options": {
                "A": entry["choice_a"],
                "B": entry["choice_b"],
                "C": entry["choice_c"],
                "D": entry["choice_d"],
            },
            "answer": entry["answer"],
            "image_path": str(os.path.basename(out_image_path))
        })
        
        # copy only if not already present
        if not out_image_path.exists():
            shutil.copy(inp_image_path, out_image_path)
        
    print(f"Extracted images to {args.output_dir}")
    # serialize dictionary to ann.json in output directory
    ann_file = args.output_dir / "seedbench_ann.json"
    with open(ann_file, 'w') as f:
        json.dump(ann_list, f, indent=4)
    print("Annotation JSON files created.")
    
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path
import base64
import os
from tqdm import tqdm
import json
import re


def write_image(payload, out_path):
    out_path = Path(out_path)

    def sniff_fmt(b: bytes):
        if b.startswith(b'\xff\xd8\xff'): return 'jpg'
        if b.startswith(b'\x89PNG\r\n\x1a\n'): return 'png'
        if b[:4] == b'RIFF' and b[8:12] == b'WEBP': return 'webp'
        if b[:6] in (b'GIF87a', b'GIF89a'): return 'gif'
        if b[:2] == b'BM': return 'bmp'
        if b[:4] in (b'II*\x00', b'MM\x00*'): return 'tiff'
        return None

    def to_bytes(x):
        # If already bytes
        if isinstance(x, (bytes, bytearray)):
            b = bytes(x)
            # If starts with data: header (rare but possible in bytes)
            if b.startswith(b'data:image'):
                b = re.sub(rb'^data:image/[^;]+;base64,', b'', b).strip()
                b = re.sub(rb'\s+', b'', b)
                return base64.b64decode(b, validate=False)
            # If it already looks like an image, just return
            if sniff_fmt(b): 
                return b
            # Try base64 decode (if someone stored ASCII base64 in bytes)
            try:
                return base64.b64decode(b, validate=True)
            except Exception:
                return b  # treat as raw bytes
        # If it's a string
        elif isinstance(x, str):
            s = x.strip()
            s = re.sub(r'^data:image/[^;]+;base64,', '', s)
            s = re.sub(r'\s+', '', s)
            try:
                return base64.b64decode(s, validate=True)
            except Exception:
                # Not base64; last resort encode (unlikely)
                return s.encode('latin1')
        else:
            raise TypeError(f"Unsupported type: {type(x)}")

    raw = to_bytes(payload)

    # Fix the "$RIFF" case (a stray leading '$' before a WebP RIFF header)
    if raw.startswith(b'$RIFF'):
        raw = raw[1:]

    fmt = sniff_fmt(raw)

    # If caller gave a wrong/missing extension, correct it
    if fmt:
        if out_path.suffix.lower() != f'.{fmt}':
            out_path = out_path.with_suffix(f'.{fmt}')
    else:
        # Unknown magic; keep user suffix or default to .bin
        if not out_path.suffix:
            out_path = out_path.with_suffix('.bin')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Read and display Parquet file contents.")
    parser.add_argument("-i", "--parquet_file", type=Path, nargs='+', help="Path to the Parquet file")
    parser.add_argument("-o", "--output_dir", type=Path, help="Directory to save extracted images", required=True)
    parser.add_argument("-s", "--split", type=str, choices=['dev', 'test'], default='dev', help="Dataset split to process")
    args = parser.parse_args()
    
    # make output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Read Parquet file(s)
    # if more than one file is given, concatenate them
    if len(args.parquet_file) > 1:
        print(f"Reading multiple Parquet files: {args.parquet_file} ...")
        df_list = [pd.read_parquet(pf) for pf in args.parquet_file]
        df = pd.concat(df_list, ignore_index=True)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {len(args.parquet_file)} files")
    else:
        print(f"Reading single Parquet file: {args.parquet_file} ...")
        df = pd.read_parquet(args.parquet_file)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    num_zeros = len(str(len(df))) + 1
    ann_dict = {}

    # columns are: index, question, A, B, C, D (options), answer, category, image (encoded like as base64-encoded JPEG (JFIF))
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        category = row['category'] if args.split == 'dev' else 'VMCBench-Test'
        # if category directory doesn't exist, create it
        category_dir = args.output_dir / category
        os.makedirs(category_dir / "images", exist_ok=True)
        # add category to ann_dict if not already present
        if category not in ann_dict:
            ann_dict[category] = []

        # zero-pad index
        filled_idx = str(idx).zfill(num_zeros)
        # write image to file
        image_filename = f"{filled_idx}"
        image_path = category_dir / "images" / image_filename
        out_image_path = write_image(row['image']['bytes'], image_path)
            
        # write question, options and answer to ann_dict
        ann_dict[category].append({
            "index": idx if args.split == 'dev' else row['index'], # in test split use original index
            "question": row['question'],
            "options": {
                "A": row['A'],
                "B": row['B'],
                "C": row['C'],
                "D": row['D']
            },
            "answer": row['answer'],
            "image_path": str(os.path.basename(out_image_path))
        })
        
    print(f"Extracted images and annotations to {args.output_dir}")
    # serialize dictionary to ann.json in each category directory
    for category, items in ann_dict.items():
        category_dir = args.output_dir / category
        ann_path = category_dir / f"{category}_ann.json"
        with open(ann_path, "w") as ann_file:
            json.dump(items, ann_file, indent=4)
    print("Annotation JSON files created.")

        
if __name__ == "__main__":
    main()

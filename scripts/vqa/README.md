# VQA Evaluation Datasets

This guide describes how to download and preprocess each VQA evaluation dataset. After processing, your directory structure should look like:

```
HAC/datasets/VQA
├── AI2D
│   └── images
├── A-OKVQA
│   └── images
├── MMSTAR
│   └── images
├── RealWorldQA
│   └── images
├── ScienceQA
│   └── images
└── SEED-Bench
    └── images
```

## Dataset Sources

Download each raw dataset from Hugging Face:

| Dataset | Link |
|---------|------|
| AI2D | [lmms-lab/ai2d](https://huggingface.co/datasets/lmms-lab/ai2d) |
| A-OKVQA | [HuggingFaceM4/A-OKVQA](https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA) |
| MMStar | [Lin-Chen/MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar) |
| RealWorldQA | [xai-org/RealworldQA](https://huggingface.co/datasets/xai-org/RealworldQA) |
| ScienceQA | [derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) |
| SEED-Bench | [lmms-lab/SEED-Bench](https://huggingface.co/datasets/lmms-lab/SEED-Bench) |

## Preprocessing

Each dataset has a corresponding processing script under `scripts/vqa/`. The general usage is:

```bash
python scripts/vqa/<dataset_name>.py -i <input_file> -o datasets/VQA/<DatasetFolder>
```

where `<input_file>` is the downloaded Parquet or JSON file and `<DatasetFolder>` is the target directory listed in the structure above.

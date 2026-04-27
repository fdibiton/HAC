from hac.config import LazyCall as L
from hac.evaluation.vqa import VQAEvaluator


evaluator = L(VQAEvaluator)(
    datasets=[
            "A-OKVQA",
            "MMSTAR",
            "SEED-Bench",
            "ScienceQA",
            "RealWorldQA",
            "AI2D"
        ],
    data_dir="datasets/VQA/",
    image_size=224,
    use_qa_prompts=False,
)
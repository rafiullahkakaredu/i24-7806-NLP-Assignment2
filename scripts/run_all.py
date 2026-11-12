"""Run the whole assignment as plain Python (same code as the notebook), e.g.

    python scripts/run_all.py --data_dir /path/to/assignment1_files
"""
import argparse, os, sys, runpy
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default=".", help="folder with cleaned.txt, raw.txt, Metadata.json")
ap.add_argument("--out_dir", default=None, help="output folder (default ./i24-7806_Assignment2_DS-X)")
ap.add_argument("--fast", action="store_true", help="smoke test with very few epochs")
a = ap.parse_args()
os.environ["A2_DATA_DIR"] = a.data_dir
if a.out_dir: os.environ["A2_OUT_DIR"] = a.out_dir
if a.fast: os.environ["A2_FAST"] = "1"
here = os.path.dirname(os.path.abspath(__file__))
g = {"__name__": "a2_pipeline"}
for part in ["a2_part0_setup.py", "a2_part1_embeddings.py", "a2_part2_sequence_labeling.py", "a2_part3_transformer.py", "a2_part4_report.py"]:
    print(f"\n{'=' * 30} {part} {'=' * 30}")
    exec(compile(open(os.path.join(here, part), encoding="utf-8").read(), part, "exec"), g)

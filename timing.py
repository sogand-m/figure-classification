#!/usr/bin/env python3
"""
Measure how long each model takes per figure.

Runs each model over the first N images, after one warm-up call so the
model-loading time is not counted. Prints seconds per figure and the
projected time for the full evaluation set and for the full 6342-figure
Torchiano corpus.

Run it from the folder that holds labels.csv:   python3 timing.py
"""

import time
import pandas as pd
import ollama

PROMPT = "Does this figure show quantitative data (a chart or plot with numbers, axes, bars, lines, points)? Answer with one word only: Q or NonQ."
MODELS = ["llava", "qwen2.5vl:7b"]
N = 10                      # figures to time per model
IMAGE_DIR = "../Figures_ESEM2022"

labels = pd.read_csv("labels.csv")
images = labels["Image"].dropna().head(N).tolist()

print(f"timing {len(images)} figures per model\n")

for model in MODELS:
    def ask(img):
        return ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": PROMPT,
                "images": [f"{IMAGE_DIR}/{img}"],
            }],
            options={"num_ctx": 16384},
        )

    # warm-up: loads the model into memory, not counted
    try:
        ask(images[0])
    except Exception as e:
        print(f"{model}: warm-up failed — {e}\n")
        continue

    start = time.time()
    ok = 0
    for img in images:
        try:
            ask(img)
            ok += 1
        except Exception:
            pass
    elapsed = time.time() - start

    if ok == 0:
        print(f"{model}: every call failed\n")
        continue

    per = elapsed / ok
    print(f"{model}")
    print(f"  {ok} figures in {elapsed:.1f}s  ->  {per:.2f} s/figure")
    print(f"  112 figures  ->  {per*112/60:.1f} min")
    print(f"  6342 figures ->  {per*6342/3600:.1f} hours")
    print()

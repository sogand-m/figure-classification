import time
import pandas as pd
import ollama

PROMPT = """What type of chart is this? Choose exactly one from this list:

Alluvial: flows between stages, band width shows amount (Sankey)
Area: line chart with the area below the line filled
Bar: bars showing one value per category (also histograms)
Bar Grouped: several bars side by side for each category
Bar Stacked: bars split into stacked segments
Bar Stacked Diverging: stacked bars going left and right from a centre line (Likert)
Beeswarm: individual dots spread so they do not overlap
Boxplot: boxes with median, quartiles and whiskers
Bubble: scatter plot where dot size shows a third value
Bump: lines showing how rankings change over time
Dendrogram: tree showing hierarchical clustering
Donut: pie chart with a hole in the middle
Dot: dots on one axis instead of bars
Forest: estimates with confidence intervals, one per row
Frequency: smooth curve showing how values are distributed (density plot)
Heatmap: grid of cells coloured by value
Interval: ranges or error bars showing an interval for each item
Line: points connected by lines, usually over time
Pie: circle split into slices
Radar: values on axes arranged in a circle (spider chart)
Scatter: dots placed by two numeric values
Slope: lines connecting two values between two points in time
Sunburst: rings split into slices, showing a hierarchy
Surface: 3D surface
Treemap: nested rectangles sized by value
Venn: overlapping circles showing sets
Violin: mirrored density shapes, one per group

Answer with only the type name, nothing else."""

MODEL = "qwen2.5vl:7b"
OUTFILE = "qwen_type_v1.csv"

FOLDER = "../Figures_ESEM2022/"
WARMUP = FOLDER + "3544902.3546245_p2_fig1.png"   # a NonQ figure, only to load the model

labels = pd.read_csv("q_labels.csv")


def ask(image):
    start = time.perf_counter()
    res = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT, "images": [image]}],
        options={"num_ctx": 16384},
    )
    seconds = time.perf_counter() - start
    return res["message"]["content"].strip(), seconds


# warm-up: load the model so loading time is not counted
print("loading", MODEL, "...")
ask(WARMUP)

rows = []
for i, (image, true_type) in enumerate(zip(labels["Image"], labels["Type"]), start=1):
    try:
        answer, seconds = ask(FOLDER + image)
    except Exception as e:
        answer, seconds = "ERROR", None
        print("  failed:", e)

    rows.append({"Image": image, "Type": true_type, "Answer": answer, "Seconds": seconds})
    print(f"{i}/{len(labels)}  {image}  true={true_type}  said={answer}  {seconds or 0:.1f} s")

    # save after every figure, so a crash loses nothing
    pd.DataFrame(rows).to_csv(OUTFILE, index=False)

# time summary
t = pd.DataFrame(rows)["Seconds"].dropna()
print(f"\n{MODEL}: mean {t.mean():.1f} s, min {t.min():.1f} s, max {t.max():.1f} s, "
      f"total {t.sum() / 60:.1f} min")
print("saved", OUTFILE)
# Phase 2 — Q chart type classification

**Question:** given a figure already known to be quantitative (Q), can its
chart type (Bar, Line, Pie, Boxplot …) be named automatically, using the
taxonomy of the reference study?

> **Summary.** On 66 quantitative figures and 27 possible chart types,
> **qwen2.5vl reaches 63.6% (macro-F1 0.579)**, against 39.4% for the SigLIP 2
> linear probe and 18.2% for llava — which is exactly the majority-class
> baseline. qwen's largest weakness is a single bias: **it never answers "Bar"**,
> and calls every simple bar chart "Bar Grouped".

## Setup

- **Corpus:** 24 ESEM 2022 papers, as in Phase 1.
- **Scope:** only quantitative figures. Non-quantitative subtypes (Code, Schema,
  Screenshot …) are outside the scope of this thesis, because the final goal —
  defect detection — applies only to charts.
- **Ground truth:** the corrected labels from Phase 1 (`labels.csv`), restricted
  to Q figures and mapped onto the type names of the reference paper
  (`make_q_labels.py` → `q_labels.csv`).
- **Evaluation set: 66 Q figures, 16 chart types.**
  Majority-class baseline (always answer "Bar"): **18.2%** (12/66).

Section 1 describes how the ground truth was prepared. Section 2 describes the
zero-shot vision-language models, section 3 the frozen encoder + linear probe,
and section 4 compares the three.

---

# 1. Preparing the ground truth

## Taxonomy

All type names come from Torchiano & Laudadio, *"Exploring How SE Academic
Papers Use Figures"* (WSESE 2025).

- **Table IV** of the paper lists **25 Q types**: Alluvial, Area, Bar,
  Bar Grouped, Bar Stacked, Bar Stacked Diverging, Beeswarm, Boxplot, Bubble,
  Bump, Dendrogram, Donut, Dot, Forest, Heatmap, Line, Pie, Radar, Scatter,
  Slope, Sunburst, Surface, Treemap, Venn, Violin.
- **Fig. 7** of the paper, which counts the Q types over all 6342 figures, uses
  two further names that do not appear in Table IV: **Frequency** (28 figures)
  and **Interval** (15 figures). The paper does not define them.
- The paper also states that **histograms are counted as Bar**, and that for
  mixed charts (e.g. bars + lines) only the **main type** is recorded.

The type list used in this phase is therefore **27 types**: the 25 of Table IV
plus Frequency and Interval.

**Open question for the supervisor:** the exact meaning of Frequency and
Interval. The definitions used in the prompt are working assumptions:
Frequency = a density curve showing how values are distributed; Interval =
ranges or error bars, one per item.

## Mapping the labels onto the taxonomy

The corrected Phase 1 labels contain **69 Q figures** under 19 type names.
Five of those names are not in the paper's taxonomy. Each case was resolved by
opening the image; the mapping is done in code, so `labels.csv` itself is not
modified.

| Original type | Figures | Mapped to | Reason |
|---|---|---|---|
| Bar Stacked Divergent | 2 | Bar Stacked Diverging | spelling, to match the paper |
| Funnel chart | 1 | Alluvial | a Sankey flow: arrow width encodes paper counts |
| Bar + Line | 1 | Bar | dual-axis bars and lines; the paper records only the main type |
| Distribuzioni | 6 | Frequency | all six are density (KDE) curves |
| Network graph | 2 | *dropped* | radial theme maps whose node size encodes counts; no type in the paper fits |

One further row was removed: `3546233_p7_fig3.png`, the bad crop already
dropped in Phase 1. It was still listed in `labels.csv` but the image no longer
exists, so the first llava run returned an error for it. This leaves
**66 figures**.

## Image fixes

Opening every figure in the five cases above also showed extraction problems:

- **Two crops contained two figures each** (`3546256_p7_fig4`,
  `3546256_p8_fig6`). Both were re-cropped by hand to the single figure.
- **Two crops were cut off** (`3546239_p4_fig2`, `3546247_p7_fig1`). Both were
  re-cropped; they still carry a black bar and scrollbars from the screen
  capture.
- `3546245_p4_fig8.png` is a byte-identical copy of `3546245_p5_fig8.png`
  (a NonQ picture). It is not in the label file, so it has no effect.

## The evaluation set

| Type | Figures | Share |
|---|---|---|
| Bar | 12 | 18.2% |
| Line | 11 | 16.7% |
| Pie | 7 | 10.6% |
| Frequency | 6 | 9.1% |
| Bar Grouped | 5 | 7.6% |
| Violin | 5 | 7.6% |
| Area | 4 | 6.1% |
| Boxplot | 3 | 4.5% |
| Bar Stacked | 3 | 4.5% |
| Alluvial | 3 | 4.5% |
| Bump | 2 | 3.0% |
| Bar Stacked Diverging | 1 | 1.5% |
| Forest | 1 | 1.5% |
| Scatter | 1 | 1.5% |
| Bubble | 1 | 1.5% |
| Heatmap | 1 | 1.5% |
| **Total** | **66** | |

Two properties of this set shape how the results must be read:

- **11 of the 27 types have no example at all** (Beeswarm, Dendrogram, Donut,
  Dot, Interval, Radar, Slope, Sunburst, Surface, Treemap, Venn). They stay in
  the prompt anyway: the model must not be told which answers are possible,
  otherwise the task becomes easier than it is in real use.
- **The distribution is long-tailed.** Five types have a single example, so a
  per-type score for those is either 0% or 100% and carries no information.
  Overall accuracy alone would also hide failures on the rare types, which is
  why macro-F1 is reported alongside it.

---

# 2. Zero-shot vision-language models

## Method

Same approach as Phase 1. Each figure is sent to a vision-language model running
locally through Ollama, one image at a time, with a single instruction. The
models are used **zero-shot**: no fine-tuning, no training data, no examples in
the prompt.

**Models.** `llava` (7B) and `qwen2.5vl:7b` (7B), with `num_ctx = 16384`
(required by qwen, see Phase 1).

**Prompt v1 — type list with definitions:**
*"What type of chart is this? Choose exactly one from this list:"*, followed by
the 27 type names, each with a one-line definition, and ending with
*"Answer with only the type name, nothing else."* The full prompt is in
`classify_q.py`.

Definitions were included because a bare list of names leaves the model to
guess what terms such as "Bump", "Forest" or "Alluvial" mean; published
ablations report the class definitions as the most important prompt component
for this kind of task.

**Scoring.** Answers are normalised (trailing full stop, letter case) and
compared with the ground truth. Reported: accuracy, macro-F1, per-type
precision and recall, and the confusion table.

## Results

| # | Model | Prompt | Accuracy | Macro-F1 |
|---|---|---|---|---|
| — | majority class | always answer "Bar" | 18.2% | — |
| 1 | llava 7B | v1 — list with definitions | **18.2%** (12/66) | **0.107** |
| 2 | qwen2.5vl 7B | v1 — list with definitions | **63.6%** (42/66) | **0.579** |

### Pilot (one figure)

Before the full run, both models were tested on a single figure
(`3546245_p4_fig2.png`, a simple bar chart with one series and nine bars):

| Model | Answer | Correct |
|---|---|---|
| llava 7B | Bar | yes |
| qwen2.5vl 7B | Bar Grouped | no |

Both followed the one-word instruction. A single figure supports no conclusion
about which model is better.

### Per-type results — llava v1

| Type | Correct | Total | Recall |
|---|---|---|---|
| Bar | 6 | 12 | 0.50 |
| Line | 2 | 11 | 0.18 |
| Pie | 2 | 7 | 0.29 |
| Frequency | 0 | 6 | 0.00 |
| Bar Grouped | 0 | 5 | 0.00 |
| Violin | 0 | 5 | 0.00 |
| Area | 1 | 4 | 0.25 |
| Boxplot | 0 | 3 | 0.00 |
| Bar Stacked | 0 | 3 | 0.00 |
| Alluvial | 0 | 3 | 0.00 |
| Bump | 0 | 2 | 0.00 |
| Bar Stacked Diverging | 0 | 1 | 0.00 |
| Forest | 1 | 1 | 1.00 |
| Scatter | 0 | 1 | 0.00 |
| Bubble | 0 | 1 | 0.00 |
| Heatmap | 0 | 1 | 0.00 |
| **All** | **12** | **66** | **18.2%** |

Only 5 of the 16 types are ever recognised. Forest's 1/1 is a single figure and
says nothing on its own.

### What llava answered

| Answer | Times given |
|---|---|
| Bar | 16 |
| Area | 12 |
| Line | 8 |
| Bar Grouped | 7 |
| Radar | 6 |
| Forest, Dot | 3 each |
| Pie, Scatter | 2 each |
| Donut, Venn, Interval | 1 each |
| *not a type name* | 4 |

**Answers that are not a type name (4 of 66):** "Poster", "Forecast",
"Sankey", and the full definition line of Bar copied back from the prompt.
"Sankey" describes the right chart (an Alluvial) under the wrong name; under
strict scoring it counts as wrong.

### Per-type results — qwen2.5vl v1

| Type | Correct | Total | Recall |
|---|---|---|---|
| Bar | 0 | 12 | 0.00 |
| Line | 10 | 11 | 0.91 |
| Pie | 6 | 7 | 0.86 |
| Frequency | 6 | 6 | 1.00 |
| Bar Grouped | 5 | 5 | 1.00 |
| Violin | 4 | 5 | 0.80 |
| Area | 0 | 4 | 0.00 |
| Boxplot | 2 | 3 | 0.67 |
| Bar Stacked | 3 | 3 | 1.00 |
| Alluvial | 2 | 3 | 0.67 |
| Bump | 0 | 2 | 0.00 |
| Bar Stacked Diverging | 1 | 1 | 1.00 |
| Forest | 1 | 1 | 1.00 |
| Scatter | 1 | 1 | 1.00 |
| Bubble | 0 | 1 | 0.00 |
| Heatmap | 1 | 1 | 1.00 |
| **All** | **42** | **66** | **63.6%** |

### What qwen answered

| Answer | Times given |
|---|---|
| Bar Grouped | 18 |
| Line | 10 |
| Pie, Frequency | 6 each |
| Alluvial, Violin | 4 each |
| Boxplot, Bar Stacked, Bar Stacked Diverging | 3 each |
| Forest, Heatmap | 2 each |
| Donut, Venn, Bubble, Scatter | 1 each |
| **Bar** | **0** |
| *not a type name* | 1 ("Sankey") |

### Where qwen's errors go (24 wrong figures)

| True type | Answered as |
|---|---|
| Bar (12) | Bar Grouped 10, Bar Stacked Diverging 2 |
| Area (4) | Bar Grouped 2, Venn 1, Bubble 1 |
| Bump (2) | Alluvial 2 |
| Alluvial (1) | "Sankey" 1 |
| Boxplot (1) | Bar Grouped 1 |
| Bubble (1) | Heatmap 1 |
| Line (1) | Forest 1 |
| Pie (1) | Donut 1 |
| Violin (1) | Boxplot 1 |

## Findings

### 2.1 llava cannot do this task

llava's accuracy (18.2%) equals the majority-class baseline, and its macro-F1
(0.107) shows that it recognises almost no type other than Bar. Its answers
collapse onto a few names: **Bar and Area alone make up 28 of its 66 answers**,
while 11 of the 16 true types are never named correctly. "Area" in particular
is given 12 times but is right only once, which suggests llava uses it as a
default for any chart with lines or filled regions.

This matches Phase 1, where llava was also the weaker model (76.1% vs 96.4% on
Q / NonQ). The gap is much larger here: telling a chart from a diagram is a
coarse visual decision, while naming one of 27 chart types needs fine detail —
bar arrangement, box and whisker shapes, density curves — that a 7B model with
a low-resolution image encoder does not resolve.

### 2.2 qwen is much better, but never answers "Bar"

qwen reaches 63.6% and a macro-F1 of 0.579, against 18.2% and 0.107 for llava
with the identical prompt. It recognises most types reliably — Frequency 6/6,
Bar Grouped 5/5, Bar Stacked 3/3, Line 10/11, Pie 6/7 — and gives a valid type
name in 65 of 66 answers (the exception, "Sankey", names the right chart under
the wrong word).

Its errors are concentrated. **qwen never answers "Bar"** — not once in 66
answers — and 10 of the 12 simple bar charts are called "Bar Grouped". This one
bias accounts for almost half of its errors. Counting those 10 as correct would
lift accuracy to about 79%, which shows how much of the remaining gap is a
single, systematic confusion rather than general weakness.

Other errors are between close neighbours: Bump called Alluvial (both show flows
between columns), Violin called Boxplot (the violin contains a boxplot), a plain
Pie called Donut.

## Run time

Measured on all 66 figures per model, on the same machine (Apple Silicon laptop,
models served by Ollama), after one warm-up call on a figure outside the
evaluation set so that model loading is not counted.

| Model | Mean per figure | Min | Max | 66 figures | 6342 figures (projected) |
|---|---|---|---|---|---|
| llava 7B | 11.9 s | 10.5 s | 16.1 s | ~13 min | ~21 hours |
| qwen2.5vl 7B | 46.4 s | 21.2 s | 115.6 s | 51.1 min | ~82 hours (3.4 days) |

**llava** took 11.9 s per figure on average (median 11.2 s). These figures are
computed over 58 of the 66 figures: the llava run was started, stopped and
restarted, and for the first 8 figures of the second run Ollama reused the
input it had already processed in the first attempt, so they took only
0.3–0.6 s. Those 8 times are excluded; the 66-figure total (~13 min) is
estimated from the mean.

**qwen** took 46.4 s per figure on average (median 39.6 s), about 4× slower
than llava and somewhat faster than the ~63 s measured in Phase 1 for the
Q / NonQ question. The wide range (21–116 s) follows image size: larger images
are encoded into more tokens.

## Practical notes

- **Timing a repeated request is misleading.** In a first test the same image
  and prompt were sent twice in a row. The second call took 0.4 s (llava) and
  0.6 s (qwen), against 16.6 s and 97.2 s for the first. Ollama reuses the
  already-processed input when a request repeats the previous one, so only the
  one-word answer is generated. The warm-up call must therefore use a different
  image from the ones being timed.
- **Robustness.** As in Phase 1, the script writes its CSV after every figure
  and catches per-image exceptions, so a failure partway through does not lose
  earlier answers.
- **One results file per model and prompt** (`llava_type_v1.csv`,
  `qwen_type_v1.csv`), never overwritten, so answers can be re-scored if the
  labels change again.

---

# 3. Frozen encoder + linear probe

## Method

The same probe as in Phase 1. Only the label changes: the chart type instead
of Q / NonQ.

1. The frozen SigLIP 2 encoder (`google/siglip2-base-patch16-224`, ~90M
   parameters) turns each of the 66 figures into a 768-dimensional embedding,
   scaled to length 1.
2. `StandardScaler` → `LogisticRegression(C=1.0)`, choosing between the 16
   types of the evaluation set. `C = 1.0` is the Phase 1 value, fixed in
   advance and not tuned.
3. Validation: **leave-one-paper-out** (20 papers → 20 rounds). In each round
   one paper's figures are the test set and the probe is trained on the other
   19, so every figure is predicted by a probe that has never seen its paper.

Run in Google Colab on a T4 GPU (`linear_probe.ipynb`, Phase 2 cells).

**Ceiling.** 7 of the 16 types appear in only one paper (Frequency, Bump, Bar
Stacked Diverging, Bubble, Forest, Heatmap, Scatter — 13 figures). When that
paper is tested, the probe has never seen the type, so the highest possible
score is **53/66 = 80.3%**.

## Results

| Method | Accuracy | 95% CI | Macro-F1 |
|---|---|---|---|
| always answer "Bar" | 18.2% (12/66) | | — |
| **SigLIP 2 + logistic probe, leave-one-paper-out** | **39.4%** (26/66) | 28.5–51.5% | **0.118** |
| same, balanced class weights | 39.4% (26/66) | 28.5–51.5% | 0.118 |
| same probe, ungrouped 5-fold CV (mean of 10 splits) | 69.5% (min 66.7%, max 72.7%) | | 0.409 |
| ceiling under leave-one-paper-out | 80.3% (53/66) | | |

On the 53 figures whose type appears in at least two papers, the probe gets
**26/53 = 49.1%**.

**The leave-one-paper-out row is the one to report.** The ungrouped row lets
the probe see figures from the same paper in training and testing; the 30-point
gap between the two measures how much that leakage is worth.

### Per-type results

| Type | Figures | Papers | Leave-one-paper-out | Ungrouped (one split) |
|---|---|---|---|---|
| Bar | 12 | 8 | **12** | 12 |
| Line | 11 | 8 | **11** | 10 |
| Pie | 7 | 2 | 0 | 6 |
| Frequency | 6 | 1 | 0 | 6 |
| Violin | 5 | 2 | 0 | 5 |
| Bar Grouped | 5 | 2 | 0 | 5 |
| Area | 4 | 3 | 3 | 2 |
| Alluvial | 3 | 3 | 0 | 0 |
| Boxplot | 3 | 3 | 0 | 0 |
| Bar Stacked | 3 | 3 | 0 | 0 |
| Bump | 2 | 1 | 0 | 2 |
| Bar Stacked Diverging | 1 | 1 | 0 | 0 |
| Bubble | 1 | 1 | 0 | 0 |
| Forest | 1 | 1 | 0 | 0 |
| Heatmap | 1 | 1 | 0 | 0 |
| Scatter | 1 | 1 | 0 | 0 |
| **All** | **66** | **20** | **26** | 48 |

### What the probe answered (leave-one-paper-out)

| Answer | Times given | Correct |
|---|---|---|
| Bar | 26 | 12 |
| Line | 26 | 11 |
| Pie | 5 | 0 |
| Area | 5 | 3 |
| Bar Stacked | 2 | 0 |
| Bar Grouped | 2 | 0 |

### Where the errors go (leave-one-paper-out, 40 wrong figures)

| True type | Answered as |
|---|---|
| Frequency (6) | Line 6 |
| Bar Grouped (5) | Bar 5 |
| Bar Stacked (3) | Bar 3 |
| Pie (7) | Bar 4, Area 2, Bar Grouped 1 |
| Violin (5) | Line 2, Pie 2, Bar Stacked 1 |
| Alluvial (3) | Line 2, Bar 1 |
| Boxplot (3) | Line 2, Bar 1 |
| Bump (2) | Pie 2 |
| Area (1) | Pie 1 |
| Bar Stacked Diverging (1) | Bar Stacked 1 |
| Bubble (1) | Bar Grouped 1 |
| Forest, Heatmap, Scatter (1 each) | Line 1 each |

## Findings

### 3.1 The probe learns the paper, not the chart type

Tested on papers it has never seen, the probe reaches 39.4%. When figures from
the same paper may appear in both training and test, it reaches 69.5%: a
**30-point gap**, against about 3 points in Phase 1 (94.3% vs 97.5%).

The types that jump from almost nothing to almost all correct are exactly the
types whose figures come from only one or two papers:

| Type | Papers | Leave-one-paper-out | Ungrouped |
|---|---|---|---|
| Pie | 2 | 0 / 7 | 6 / 7 |
| Frequency | 1 | 0 / 6 | 6 / 6 |
| Violin | 2 | 0 / 5 | 5 / 5 |
| Bar Grouped | 2 | 0 / 5 | 5 / 5 |
| Bump | 1 | 0 / 2 | 2 / 2 |

Pie is the clearest case. A pie looks nothing like a bar chart, yet the probe
finds none of the seven pies when their paper is unseen, and six of them when
other figures from the same paper are in training. In the grouped run it also
answered "Pie" five times, never on a pie.

The explanation is that every paper has its own visual style — colours, fonts,
background, layout — and the SigLIP embedding captures that style strongly. With
most types concentrated in one or two papers, the probe can score well by
recognising the paper instead of the chart type. In Phase 1 this did not help:
Q and NonQ figures occur in almost every paper, so the style of a paper says
little about the label.

### 3.2 Class imbalance is not the cause

Re-training with balanced class weights, so that every type counts equally,
gives exactly the same result (26/66). The problem is not that Bar and Line are
frequent, but that the other types are seen in too few papers. *(This check was
added after seeing the first result.)*

### 3.3 What would help

The limit is the number of **papers** per type, not the encoder. The only types
the probe learns are Bar and Line, which come from 8 papers each. More data
from many papers — such as the images behind `SEVizData.csv` — is what the
probe needs; a larger encoder alone is unlikely to fix this.

## Run time

Not re-measured in this phase: the time per figure is the same as in Phase 1.
Almost all of it is spent in the encoder, which does the same work whatever the
task, and every image is resized to 224 × 224 pixels first. The classifier now
chooses between 16 types instead of 2, which still takes far below 1 ms.

Measured in Phase 1 (`linear_probe.ipynb`, Google Colab, T4 GPU, 122 figures,
after one warm-up call):

| Part | Time |
|---|---|
| Encoder (image → embedding) | 43.8 ms per figure |
| Training the classifier | 70 ms, once |
| Classifier answer | 0.043 ms per figure |
| **66 figures** | **~3 s** |
| **6342 figures (projected)** | **~4.6 min** |

The encoder time includes reading each image from Google Drive, so the encoder
alone is somewhat faster.

---

# 4. Comparing the three approaches

| Model | Approach | Accuracy | Macro-F1 | Parameters | Machine | Time per figure | 6342 figures (projected) |
|---|---|---|---|---|---|---|---|
| llava 7B | zero-shot VLM | 18.2% (12/66) | 0.107 | ~7B | laptop (Ollama) | 11.9 s | ~21 hours |
| **qwen2.5vl 7B** | zero-shot VLM | **63.6%** (42/66) | **0.579** | ~7B | laptop (Ollama) | 46.4 s | ~82 hours (3.4 days) |
| `google/siglip2-base-patch16-224` + logistic regression | frozen encoder + linear probe, leave-one-paper-out | 39.4% (26/66) | 0.118 | ~90M (~12k trained) | Colab T4 GPU | ~44 ms | ~4.6 min |

For reference, always answering "Bar" gives 18.2%.

**How to read the table:**

- **qwen is clearly the most accurate**, on both accuracy and macro-F1, but
  also the slowest.
- **The probe is about 1000× faster than qwen per figure**, but its 39.4% comes
  almost entirely from Bar and Line; its macro-F1 (0.118) is close to llava's.
- **llava is no better than always answering "Bar".**

**Caveats:**

- **Different machines.** The VLMs ran on a laptop, the probe on a Colab T4
  GPU, so the times are not a like-for-like comparison. Parameter count (~7B vs
  ~90M) is the hardware-independent comparison.
- **Different protocols.** The VLMs answer every figure zero-shot; the probe is
  trained on the other papers and tested on one paper at a time.
- **The probe has a ceiling of 80.3%**, because 7 types appear in only one
  paper; the VLMs have no such limit.

---

# Limitations

- **Small, long-tailed set:** 66 figures over 16 types, several with a single
  example.
- **Taxonomy gaps:** Frequency and Interval are undefined in the paper; charts
  with sized nodes fit no type; mixed charts are reduced to one main type.
- **Single annotator** for the original labels, with the Phase 1 corrections
  also made by one person.
- **No development / test split:** the prompt is written and scored on the same
  figures, so results are mildly optimistic.
- **Few papers per type (probe):** 7 of the 16 types come from a single paper,
  so under leave-one-paper-out the probe never sees them in training and cannot
  score above 80.3% (53/66).
- **Different machines for timing:** the VLMs ran on an Apple Silicon laptop
  through Ollama, the probe on a Google Colab T4 GPU, so the run times are not
  directly comparable. Parameter count (~7B vs ~90M) is the hardware-independent
  comparison.

# Appendix — file formats

- `q_labels.csv` is an ordinary comma-separated file with columns
  `Image, Paper, Fig, Type, Label`. It is generated by `make_q_labels.py` and
  can be re-created at any time.
- Each results file (`llava_type_v1.csv`, `qwen_type_v1.csv`) has columns
  `Image, Type, Answer, Seconds`: the figure, the correct type, the model's raw
  answer, and the time in seconds.

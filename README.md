# HNS-Spill-Detection

A YOLOv3 object detector, implemented in **TensorFlow 1.x**, trained to localise
**HNS (Hazardous and Noxious Substances) spills** in images. The network is a
Darknet-53 backbone with a three-scale feature-pyramid head, trained with GIoU
box regression, focal-weighted objectness loss and label smoothing.

The repository covers the full pipeline: weight conversion → two-stage transfer
learning → evaluation with VOC-style mAP → graph freezing → batch inference.

---

## Repository layout

```
HNS-Spill-Detection/
├── core/                        # the model package — everything imports from here
│   ├── config.py                # single source of truth for all hyper-parameters
│   ├── common.py                # conv / residual / route / upsample building blocks
│   ├── backbone.py              # Darknet-53 feature extractor
│   ├── yolov3.py                # YOLOV3 class: network, decode, losses
│   ├── dataset.py               # Dataset iterator: augmentation + label encoding
│   └── utils.py                 # letterbox resize, NMS, box post-processing, drawing
├── data/
│   ├── anchors/basline_anchors.txt   # 9 anchors, stride-normalised
│   ├── classes/hns.names             # one class name per line  ← EDIT THIS
│   ├── dataset/                      # hns_train.txt / hns_test.txt (see its README)
│   ├── detection/                    # evaluate.py writes annotated images here
│   └── log/                          # TensorBoard summaries
├── mAP/
│   ├── main.py                  # computes AP per class + mAP, draws plots
│   ├── ground-truth/            # written by evaluate.py
│   ├── predicted/               # written by evaluate.py
│   └── images/                  # written by evaluate.py
├── checkpoint/                  # .ckpt files (git-ignored)
├── convert_weight.py            # remap a COCO YOLOv3 checkpoint onto this model
├── train.py                     # two-stage training loop
├── evaluate.py                  # run the detector over the test set, dump mAP inputs
├── freeze_graph.py              # checkpoint → frozen .pb graph
├── image_demo.py                # batch inference from a .pb graph
└── requirements.txt
```

Every module imports as `core.<module>`, so **all scripts must be run from the
repository root**.

---

## How the model works

### Backbone and head — `core/backbone.py`, `core/yolov3.py`

`darknet53()` stacks `convolutional` and `residual_block` units from
`core/common.py` (3×3 stride-2 convs for downsampling, batch-norm, LeakyReLU
α=0.1) and returns three feature maps. `YOLOV3.__build_nework` attaches an FPN
head on top: the deepest map produces `conv_lbbox`, is then 1×1-reduced,
nearest-neighbour upsampled and concatenated with `route_2` to produce
`conv_mbbox`, and the same again with `route_1` to produce `conv_sbbox`.

Each head outputs `3 * (num_classes + 5)` channels — 3 anchors per scale, each
predicting `(tx, ty, tw, th, objectness, class scores…)`.

### Decoding — `YOLOV3.decode`

Raw head output is turned into image-space boxes:

```
pred_xy = (sigmoid(tx, ty) + grid_cell) * stride
pred_wh = exp(tw, th) * anchor * stride
```

Objectness and class scores go through a sigmoid (multi-label, not softmax), so
`pred_*bbox` has shape `[batch, size, size, 3, 5 + num_classes]`.

Strides are `[8, 16, 32]` — `pred_sbbox` catches small objects, `pred_lbbox`
large ones.

### Loss — `YOLOV3.loss_layer` / `compute_loss`

Three terms, summed over all three scales:

| Term        | What it does                                                                                      |
| ----------- | ------------------------------------------------------------------------------------------------- |
| `giou_loss` | `1 - GIoU` on positive cells, weighted by `2 - (w·h / input_size²)` so small boxes count more      |
| `conf_loss` | Sigmoid cross-entropy on objectness, scaled by a focal term `\|target − pred\|²`                   |
| `prob_loss` | Sigmoid cross-entropy on class scores, positive cells only                                        |

A negative cell is only penalised if its best IoU against *any* ground-truth box
is below `IOU_LOSS_THRESH` (0.5) — this stops near-misses from being punished as
background.

### Data pipeline — `core/dataset.py`

`Dataset` is an iterator yielding
`(images, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)`.

* **Augmentation** (train only, if `DATA_AUG`): random horizontal flip, random
  crop, random translate — each applied with probability 0.5, with boxes
  transformed to match.
* **Multi-scale training**: every batch picks a random side length from
  `TRAIN.INPUT_SIZE` (320 … 608).
* **Letterboxing** (`utils.image_preporcess`): aspect-preserving resize, pad to
  square with grey (128), scale to `[0, 1]`.
* **Label encoding** (`preprocess_true_boxes`): a box is assigned to *every*
  anchor whose IoU with it exceeds 0.3; if no anchor qualifies, it falls back to
  the single best anchor across all scales. Class targets use label smoothing
  (ε = 0.01). At most 150 boxes per scale per image.

### Post-processing — `core/utils.py`

`postprocess_boxes` converts `xywh → xyxy`, undoes the letterbox padding back to
original image coordinates, clips to the image, and drops boxes whose
`objectness × class_score` is below the score threshold. `nms` then runs
per-class greedy NMS (`method='soft-nms'` is also implemented).

---

## Setup

The code is TensorFlow 1.x graph-mode (`tf.placeholder`, `tf.Session`,
`tf.variable_scope`) and **will not run on TensorFlow 2.x**. Use Python 3.6/3.7.

```bash
git clone https://github.com/kazim05/HNS-Spill-Detection.git
cd HNS-Spill-Detection
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Preparing your data

**1. Class names — `data/classes/hns.names`**

One class per line, 0-indexed. The file currently ships with **placeholder
names** (`class_0`, `class_1`, `class_2`) — replace them with your real HNS
classes. The count here defines `num_classes` everywhere; changing it changes
the head output shape, so existing checkpoints become incompatible.

**2. Annotations — `data/dataset/hns_train.txt` and `hns_test.txt`**

One line per image:

```
<image_path> <xmin>,<ymin>,<xmax>,<ymax>,<class_id> <xmin>,<ymin>,<xmax>,<ymax>,<class_id> ...
```

Absolute pixel coordinates, `class_id` matching the line number in
`hns.names`. Images with no boxes are skipped. See
[`data/dataset/README.md`](data/dataset/README.md) for a worked example.

**3. Anchors — `data/anchors/basline_anchors.txt`**

Nine `w,h` pairs on a single comma-separated line, reshaped to `(3, 3, 2)` —
three anchors per scale, **already divided by their stride**. The shipped file
is the standard COCO baseline; if you re-cluster anchors on your own data,
remember to divide each `(w, h)` by `8 / 16 / 32` for the small / medium / large
group respectively.

---

## Usage

All commands run from the repository root.

### 1. Initialise weights (optional but recommended)

Transfer learning from a COCO-pretrained YOLOv3 converges far faster than
training from scratch. Place the COCO checkpoint at the path in
`cfg.YOLO.ORIGINAL_WEIGHT` and run:

```bash
python convert_weight.py --train_from_coco
```

This matches the COCO variables to this graph's variables in order, skipping the
three detection heads (`conv_sbbox`, `conv_mbbox`, `conv_lbbox`) whose shapes
depend on `num_classes`, and saves the result to `cfg.YOLO.DEMO_WEIGHT`. It
raises `RuntimeError` if the variable counts or shapes don't line up.

Then point `cfg.TRAIN.INITIAL_WEIGHT` at that checkpoint.

### 2. Train

```bash
python train.py
```

Training runs in two stages, controlled by `config.py`:

* **Stage 1** (`FISRT_STAGE_EPOCHS`, default 70) — backbone frozen, only the
  three detection heads are optimised. This lets the new heads settle without
  destroying pretrained features.
* **Stage 2** (`SECOND_STAGE_EPOCHS`, default 80) — all variables trainable.

Learning rate warms up linearly over `WARMUP_EPOCHS`, then follows a cosine decay
from `LEARN_RATE_INIT` to `LEARN_RATE_END`. An exponential moving average
(decay 0.9995) is maintained over all trainable variables.

Outputs:

| Path                                          | Contents                                      |
| --------------------------------------------- | --------------------------------------------- |
| `checkpoint/yolov3_test_loss=<loss>.ckpt-<ep>` | one checkpoint per epoch (last 10 kept)       |
| `data/train_loss.csv`, `data/test_loss.csv`    | per-step loss, flushed live                   |
| `data/log/`                                    | TensorBoard summaries (**wiped on each run**) |

If `INITIAL_WEIGHT` cannot be restored, training falls back to scratch and skips
stage 1 automatically.

Monitor with `tensorboard --logdir data/log`.

### 3. Evaluate

Set `cfg.TEST.WEIGHT_FILE` to the checkpoint you want to score, then:

```bash
python evaluate.py          # writes mAP/ground-truth, mAP/predicted, mAP/images
cd mAP && python main.py    # computes AP per class and mAP
```

`evaluate.py` restores the **EMA** copy of the weights, runs the detector over
`cfg.TEST.ANNOT_PATH`, and writes one `.txt` per image in the format
`mAP/main.py` expects (`<class> <score> <xmin> <ymin> <xmax> <ymax>` for
predictions, without the score for ground truth). With `WRITE_IMAGE` enabled it
also saves annotated images to `data/detection/`.

`mAP/main.py` computes VOC-style AP (interpolated precision, area under the
precision-recall curve) at an IoU threshold of 0.5 and writes to `mAP/results/`:
`results.txt`, a per-class PR curve, ground-truth and prediction histograms, and
`mAP.png`. Useful flags:

```bash
python main.py -na                     # no animation window
python main.py -np                     # no plots
python main.py -q                      # quiet
python main.py -i class_1              # ignore a class
python main.py --set-class-iou class_0 0.75
```

### 4. Freeze the graph and run inference

```bash
python freeze_graph.py --ckpt_file ./checkpoint/your_best.ckpt-142 --pb_file ./yolov3_hns.pb
python image_demo.py --image_dir /path/to/images --pb_file ./yolov3_hns.pb --result_dir ./result
```

`image_demo.py` loads the frozen graph once, runs every image in `--image_dir`
through it, and writes annotated copies to `--result_dir` (which is **wiped** at
the start of each run). Score and IoU thresholds default to the `cfg.TEST`
values and can be overridden with `--score_threshold` / `--iou_threshold`.

---

## Configuration reference — `core/config.py`

Everything is read through `from core.config import cfg`.

| Key                             | Default                        | Meaning                                              |
| ------------------------------- | ------------------------------ | ---------------------------------------------------- |
| `YOLO.CLASSES`                  | `./data/classes/hns.names`     | class name list; its length is `num_classes`          |
| `YOLO.ANCHORS`                  | `./data/anchors/basline_anchors.txt` | 9 stride-normalised anchors                     |
| `YOLO.STRIDES`                  | `[8, 16, 32]`                  | downsampling factor of each detection scale           |
| `YOLO.ANCHOR_PER_SCALE`         | `3`                            | anchors per scale                                     |
| `YOLO.IOU_LOSS_THRESH`          | `0.5`                          | above this IoU a negative cell is ignored             |
| `YOLO.UPSAMPLE_METHOD`          | `resize`                       | `resize` (nearest-neighbour) or `deconv` (TensorRT)   |
| `YOLO.MOVING_AVE_DECAY`         | `0.9995`                       | EMA decay for evaluation weights                      |
| `YOLO.ORIGINAL_WEIGHT`          | `./checkpoint/yolov3_coco.ckpt`| input to `convert_weight.py`                          |
| `YOLO.DEMO_WEIGHT`              | `./checkpoint/yolov3_coco_demo.ckpt` | output of `convert_weight.py`                   |
| `TRAIN.ANNOT_PATH`              | `./data/dataset/hns_train.txt` | training annotations                                  |
| `TRAIN.BATCH_SIZE`              | `2`                            | small — tuned for limited GPU memory                  |
| `TRAIN.INPUT_SIZE`              | `[320 … 608]`                  | multi-scale training sizes                            |
| `TRAIN.DATA_AUG`                | `True`                         | flip / crop / translate                               |
| `TRAIN.LEARN_RATE_INIT` / `_END`| `1e-5` / `1e-10`               | cosine schedule endpoints                             |
| `TRAIN.WARMUP_EPOCHS`           | `2`                            | linear LR warm-up                                     |
| `TRAIN.FISRT_STAGE_EPOCHS`      | `70`                           | frozen-backbone epochs *(name is misspelled in code)* |
| `TRAIN.SECOND_STAGE_EPOCHS`     | `80`                           | full fine-tuning epochs                               |
| `TRAIN.INITIAL_WEIGHT`          | a `.ckpt-150` path             | checkpoint to resume/transfer from                    |
| `TEST.ANNOT_PATH`               | `./data/dataset/hns_test.txt`  | test annotations                                      |
| `TEST.INPUT_SIZE`               | `608`                          | fixed inference resolution                            |
| `TEST.SCORE_THRESHOLD`          | `0.3`                          | minimum class score                                   |
| `TEST.IOU_THRESHOLD`            | `0.45`                         | NMS IoU threshold                                     |
| `TEST.WEIGHT_FILE`              | a `.ckpt-142` path             | checkpoint scored by `evaluate.py`                    |
| `TEST.WRITE_IMAGE` / `_PATH`    | `True` / `./data/detection/`   | save annotated detections                             |

The `INITIAL_WEIGHT` and `WEIGHT_FILE` defaults point at checkpoints from a
previous training run and are **not** in the repository — set them to your own.

---

## Known issues and gotchas

* **`core/dataset.py:112-113`** — `random_crop` uses
  `max(w, …)` / `max(h, …)` for the crop's far edge where it should be `min`, so
  the crop is never actually bounded by the image. A fix is open on the
  `codex/find-and-fix-important-bug` branch and has not been merged.
* **`core/yolov3.py:25-28`** — the network build is wrapped in a bare
  `except:` that re-raises `NotImplementedError`, hiding the real error. Comment
  out the `try`/`except` when debugging a shape mismatch.
* **`core/dataset.py:194`** uses `np.float`, removed in NumPy ≥ 1.24. Fine with
  the pinned `numpy==1.15.1`; replace with `float` on newer NumPy.
* **`mAP/main.py`** calls `fig.canvas.set_window_title`, removed in
  Matplotlib ≥ 3.4. Use an older Matplotlib or run with `-np` to skip plotting.
* **`cfg.TEST.DATA_AUG` is `True`** — augmentation is applied to the test set
  when it is loaded through `Dataset('test')` in `train.py`. Set it to `False`
  if you want a deterministic validation loss. (`evaluate.py` does not use
  `Dataset`, so it is unaffected.)
* `train.py` deletes `data/log/` at startup, so TensorBoard history does not
  survive across runs.

---

## Credits

The YOLOv3 implementation is derived from
[YunYang1994/tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3),
adapted for the HNS spill dataset. `mAP/main.py` comes from
[Cartucho/mAP](https://github.com/Cartucho/mAP).

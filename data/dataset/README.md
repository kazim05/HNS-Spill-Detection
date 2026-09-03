# Annotation files

`core/config.py` expects two plain-text files in this folder:

| File            | Used by                                | Config key             |
| --------------- | -------------------------------------- | ---------------------- |
| `hns_train.txt` | `train.py` (via `Dataset('train')`)    | `cfg.TRAIN.ANNOT_PATH` |
| `hns_test.txt`  | `train.py`, `evaluate.py`              | `cfg.TEST.ANNOT_PATH`  |

They are **not** committed (see `.gitignore`) because the image paths are
machine-specific.

## Format

One line per image, space separated:

```
<image_path> <xmin>,<ymin>,<xmax>,<ymax>,<class_id> <xmin>,<ymin>,<xmax>,<ymax>,<class_id> ...
```

* `image_path` — path `cv2.imread` can open. No spaces in the path.
* Box coordinates are **absolute pixels** in the original image, `xmin,ymin`
  is top-left and `xmax,ymax` is bottom-right. Integers, no spaces inside a box.
* `class_id` is the 0-based line number of the class in
  `data/classes/hns.names`.
* Images with zero boxes are dropped by `Dataset.load_annotations`.

## Example

```
/data/hns/images/img_0001.jpg 108,64,342,290,0
/data/hns/images/img_0002.jpg 12,20,180,155,2 200,90,410,300,0
/data/hns/images/img_0003.jpg 55,140,300,410,1
```

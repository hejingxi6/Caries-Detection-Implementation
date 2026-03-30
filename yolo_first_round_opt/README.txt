First-round YOLOv8 optimization package for dental lesion detection

What this package does
1. train_first_round.py
   - higher resolution
   - multi-scale training
   - small-object-friendly augmentation
   - copy-paste + mild mixup
   - keeps the workflow inside standard Ultralytics YOLO

2. sweep_thresholds.py
   - sweeps confidence and NMS IoU
   - writes CSV + best config JSON
   - meant for val first, then test

3. sahi_infer.py
   - runs sliced inference for tiny lesions
   - useful when small lesion recall is weak

Assumptions
- You already have:
  - data.yaml
  - YOLO-format labels
  - Ultralytics YOLOv8 installed or installable
- Your dataset uses the standard structure referenced by data.yaml.

Install
pip install -r requirements.txt

Example workflow

Step 1: training
python train_first_round.py ^
  --data data.yaml ^
  --weights yolov8m.pt ^
  --project runs/first_round ^
  --name imgsz960_stage1 ^
  --imgsz 960 ^
  --epochs 100 ^
  --batch 8 ^
  --device 0 ^
  --pretrained

Step 2: sweep thresholds on validation set
python sweep_thresholds.py ^
  --weights runs/first_round/imgsz960_stage1/weights/best.pt ^
  --data data.yaml ^
  --imgsz 960 ^
  --batch 8 ^
  --device 0 ^
  --split val ^
  --project runs/first_round ^
  --name sweep_val_stage1

Step 3: after choosing best val operating point, run the same sweep on test
python sweep_thresholds.py ^
  --weights runs/first_round/imgsz960_stage1/weights/best.pt ^
  --data data.yaml ^
  --imgsz 960 ^
  --batch 8 ^
  --device 0 ^
  --split test ^
  --project runs/first_round ^
  --name sweep_test_stage1

Step 4: SAHI sliced inference on hard images
python sahi_infer.py ^
  --weights runs/first_round/imgsz960_stage1/weights/best.pt ^
  --source sample_images ^
  --save_dir runs/sahi_predict ^
  --confidence 0.05 ^
  --device cuda:0 ^
  --slice_height 512 ^
  --slice_width 512 ^
  --overlap_height_ratio 0.20 ^
  --overlap_width_ratio 0.20

Notes
- Do not compare runs unless:
  - same train/val/test split
  - same seed
  - same metrics
- For your first round, do not touch backbone surgery yet.
- First check whether imgsz + threshold sweep + SAHI improves the weak lesion classes.

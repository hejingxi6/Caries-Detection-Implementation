@echo off
python 03_train_stage1_binary.py ^
  --dataset_root "D:\projects\yolo_fourth_round_opt\stage1_binary" ^
  --project "D:\projects\yolo_fourth_round_opt\runs_stage1" ^
  --name "exp_fix_nan_v2" ^
  --image_size 640 ^
  --epochs 80 ^
  --batch 4 ^
  --lr 0.0001 ^
  --norm_json "D:\projects\yolo_fourth_round_opt\norm_stats.json"
pause

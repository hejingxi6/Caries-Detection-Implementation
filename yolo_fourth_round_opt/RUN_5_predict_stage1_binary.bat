@echo off
python 04_predict_stage1_binary.py ^
  --data "D:\projects\yolo_fourth_round_opt\data_stage1_fixed.yaml" ^
  --weights "D:\projects\yolo_fourth_round_opt\runs_stage1\exp_fix_nan_v2\best_stage1.pth" ^
  --out_dir "D:\projects\yolo_fourth_round_opt\stage1_pred" ^
  --threshold 0.50 ^
  --expand_ratio 0.08
pause
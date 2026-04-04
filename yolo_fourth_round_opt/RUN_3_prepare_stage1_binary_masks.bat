@echo off
python 02_prepare_stage1_binary_masks.py ^
  --data "D:\projects\yolo_fourth_round_opt\data_stage1_fixed.yaml" ^
  --out_root "D:\projects\yolo_fourth_round_opt\stage1_binary" ^
  --foreground_classes all
pause
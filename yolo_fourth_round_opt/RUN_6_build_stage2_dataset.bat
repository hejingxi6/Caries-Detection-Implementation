@echo off
python 05_build_stage2_lesion_dataset.py ^
  --data "D:\projects\yolo_fourth_round_opt\data_stage1_fixed.yaml" ^
  --roi_json "D:\projects\yolo_fourth_round_opt\stage1_pred\stage1_rois.json" ^
  --out_root "D:\projects\yolo_fourth_round_opt\stage2_lesion3_v2" ^
  --lesion_classes 0,1,2 ^
  --tooth_class 3 ^
  --expand_ratio 0.30 ^
  --norm_json "D:\projects\yolo_fourth_round_opt\norm_stats.json"
pause
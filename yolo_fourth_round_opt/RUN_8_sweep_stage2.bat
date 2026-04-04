@echo off
python 07_sweep_stage2_thresholds.py ^
  --weights "D:\projects\yolo_fourth_round_opt\runs_stage2\lesion3_v3_m960_e60\weights\best.pt" ^
  --data "D:\projects\yolo_fourth_round_opt\stage2_lesion3_v3\stage2_lesion3_v3.yaml" ^
  --imgsz 960 ^
  --batch 1 ^
  --device 0 ^
  --split val ^
  --project "D:\projects\yolo_fourth_round_opt\runs_stage2" ^
  --name "lesion3_v3_sweep_val" ^
  --confs 0.10,0.15,0.20,0.25,0.30 ^
  --ious 0.45,0.50,0.55,0.60
pause
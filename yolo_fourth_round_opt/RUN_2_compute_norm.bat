@echo off
python 01_compute_global_norm.py ^
  --data "D:\projects\yolo_third_round_opt\round3_data\data_round3.yaml" ^
  --out_json "norm_stats.json"
pause
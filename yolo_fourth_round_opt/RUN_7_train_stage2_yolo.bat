@echo off
yolo detect train ^
  model=yolov8m.pt ^
  data="D:\projects\yolo_fourth_round_opt\stage2_lesion3_v3\stage2_lesion3_v3.yaml" ^
  imgsz=960 ^
  epochs=60 ^
  batch=2 ^
  device=0 ^
  workers=4 ^
  optimizer=AdamW ^
  lr0=0.00035 ^
  close_mosaic=10 ^
  cos_lr=True ^
  cache=disk ^
  amp=True ^
  project="D:\projects\yolo_fourth_round_opt\runs_stage2" ^
  name="lesion3_v3_m960_e60"
pause
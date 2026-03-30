YOLOv8 Dental Lesion Detection - Third Round Upgrade

What this round changes
1. Stronger lesion-heavy oversampling than round 2
2. Lesion-aware crop generation from train images
3. Higher-resolution overnight training
4. Stable YOLOv8 training (multi_scale kept off)
5. Standard threshold sweep after training
6. Optional SAHI inference check after training

Why these changes
- Round 1 improved after threshold sweep, but lesion classes remained weak.
- Round 2 improved recall and Crack, but Caries was still weak and background confusion remained high.
- The deep-research scan consistently pointed to:
  higher resolution + lesion-focused sampling + tile/crop treatment + strict evaluation
  as the most practical fast-gain levers for tiny lesion detection. This package implements those ideas
  inside the same Ultralytics YOLOv8 workflow instead of switching frameworks. See the research prompt
  and summary provided by the user for the exact constraints and diagnosis.

Folder structure after unzip
yolo_third_round_opt/
├── build_round3_lesion_mix.py
├── train_round3.py
├── sweep_thresholds_round3.py
├── sahi_eval_round3.py
├── requirements.txt
└── README.txt

Recommended nightly run
Step 1: build round-3 data
python build_round3_lesion_mix.py --data D:\projects\yolov8_caries_detector_clean\yolo_dataset\data.yaml --out_dir D:\projects\yolo_third_round_opt\round3_data --repeat_map 0:3,1:2,2:6 --crop_repeat_map 0:2,1:2,2:4 --background_repeat 1 --min_crop_size 384 --max_crop_size 960 --context_scale 7.0 --max_crops_per_image 6

Step 2: overnight training
python train_round3.py --data D:\projects\yolo_third_round_opt\round3_data\data_round3.yaml --weights yolov8m.pt --project runs/third_round --name lesionmix_imgsz960 --imgsz 960 --epochs 100 --batch 2 --device 0 --pretrained

Heavier alternative (only if memory and time allow)
python train_round3.py --data D:\projects\yolo_third_round_opt\round3_data\data_round3.yaml --weights yolov8m.pt --project runs/third_round --name lesionmix_imgsz1024 --imgsz 1024 --epochs 120 --batch 2 --device 0 --pretrained

Step 3: threshold sweep
python sweep_thresholds_round3.py --weights D:\projects\yolo_third_round_opt\runs\detect\runs\third_round\lesionmix_imgsz960\weights\best.pt --data D:\projects\yolov8_caries_detector_clean\yolo_dataset\data.yaml --imgsz 960 --batch 2 --device 0 --split val --project runs/third_round --name sweep_val_lesionmix_imgsz960

Optional SAHI check after training
python sahi_eval_round3.py --weights D:\projects\yolo_third_round_opt\runs\detect\runs\third_round\lesionmix_imgsz960\weights\best.pt --source D:\projects\yolov8_caries_detector_clean\yolo_dataset\val\images --save_dir runs\sahi_round3_val --confidence 0.05 --device cuda:0

What to compare tomorrow
- Round 1 best threshold sweep vs Round 2 best threshold sweep vs Round 3 best threshold sweep
- Overall precision / recall / mAP50 / mAP50-95
- Per-class AP50 for Caries / Cavity / Crack / Tooth
- Confusion matrix: did background confusion drop?
- Did Caries improve without destroying Crack or precision?

Honest expectation
This round is designed to be materially stronger than rounds 1 and 2, but 80%+ performance is NOT guaranteed.
If your labels are noisy, lesion boxes are tiny, and classes remain extremely imbalanced, this package may improve results
substantially without reaching 0.80 mAP. It is designed for credible improvement, not fake promises.

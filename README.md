# Transfer Learning for Dental Caries Detection Using YOLOv8

## English

This repository documents a YOLOv8-based dental caries detection project as an applied case study of transfer learning. A pre-trained YOLOv8 object detector was fine-tuned on a small and imbalanced dental image dataset to examine its performance, limitations, and possible improvement directions.

This project is **not** presented as a clinically reliable diagnostic system. Instead, it is used as a machine learning case study to understand how transfer learning behaves in a small-scale medical object detection task.

## Project Positioning

The project was originally developed as a YOLO-based dental caries detection implementation and was later reorganized as a transfer learning case study.

The core idea is:

- start from a pre-trained YOLOv8 object detection model;
- fine-tune it on dental image data;
- evaluate the detector using object detection metrics;
- analyze why performance is limited under small, imbalanced, and visually difficult medical data;
- identify future improvement directions rather than overclaiming clinical reliability.

Because this is an object detection task, the main metrics are:

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95

Classification accuracy is not used as the main metric because the model needs to both locate and classify dental regions.

## Repository Structure

```text
Caries-Detection-Implementation/
├── yolo_first_round_opt/
├── yolo_second_round_opt/
├── yolo_third_round_opt/
├── yolo_fourth_round_opt/
├── report/
├── .gitignore
└── README.md

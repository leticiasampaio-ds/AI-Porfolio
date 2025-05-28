# X-Ray Baggage Scanner Anomaly Detection

## Data source
Kaggle: [X-Ray Baggage Scanner Anomaly Detection.](https://www.kaggle.com/datasets/orvile/x-ray-baggage-anomaly-detection)

## Exploratory Data Analysis
- The dataset is composed of 5 object classes, all well-represented in train/val/test splits.
    - Class 2 is less frequent across all subsets, which may lead to class imbalance during training.
- All bounding boxes follow the YOLO format with normalized coordinates.
- All images in the dataset have a fixed resolution of *416×416 pixels*

**About object size and aspect ratio**:  
- Classes vary significantly in bounding box dimensions. This suggests that anchor-based models may need tuning to better fit small vs large objects or anchor-free architectures must be considered.

## Model Selection
Based on EDA
## Project Overview

This project was developed as part of the GET 324 course, covering the full pipeline of an AI application:
dataset preparation, model training, evaluation, and cloud deployment.

- **Task**: Binary image classification (Healthy Tomato vs Tomato Mosaic Virus)
- **Model**: Custom CNN
- **Framework**: TensorFlow / Keras
- **Deployment**: Streamlit Community Cloud

## Dataset

- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) (Kaggle)
- Only two classes were used from the full dataset:
  - `Tomato_healthy`
  - `Tomato__Tomato_mosaic_virus`

**Note:** The raw dataset is not included in this repository due to size. To reproduce:
1. Download the dataset from the link above.
2. Extract the `Tomato_healthy` and `Tomato__Tomato_mosaic_virus` folders.
3. Place them inside `dataset/` following the structure below.

```
dataset/
├── plant village/
    ├── tomato_healthy
    ├──Tomato__Tomato_mosaic_virus
```

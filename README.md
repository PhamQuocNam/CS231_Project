# 🖥️ Computer Vision Fundamentals (CS231.P22)

<p align="center">
  <a href="https://www.uit.edu.vn/" title="University of Information Technology">
    <img src="./assets/img1.png" alt="University of Information Technology" width="300">
  </a>
</p>


## 📝 Project Overview

- **Project Title**: Human Detection in Special Environments (Fog, Darkness, etc.)
- **Course**: CS231 - Introduction to Computer Vision  
- **Semester**: HK2 (2024–2025)  
- **Instructor**: Mai Tiến Dũng  

## 👥 Team Members

| #  | Student ID | Name             | Role        | GitHub                                                | Email                     |
|----|------------|------------------|-------------|--------------------------------------------------------|---------------------------|
| 1  | 23520984   | Phạm Quốc Nam    | Team Leader | [@PhamQuocNam](https://github.com/PhamQuocNam)         | 23520984@gm.uit.edu.vn    |
| 2  | 23520044   | Hà Tuấn Anh      | Member      |                                                        |                           |

## 🎥 Project Demonstration

### 🔆 Normal Environment



https://github.com/user-attachments/assets/ab8cdccd-b66c-4112-9623-897c3b4460d5



### 🌫️ Special Environment (Fog / Darkness)


https://github.com/user-attachments/assets/d5254332-95b5-47f1-b457-b7ab23f5b853


## 🛠️ Installation & Usage

### ✅ Prerequisites
- Python 3.x
- Git

### 📦 Setup Instructions

```bash
git clone https://github.com/PhamQuocNam/CS231_Project.git
cd CS231_Project
pip install -r requirements.txt
```

### 📁 Project Folder Structure
```bash
CS231_Project/
├── checkpoints/      # Model checkpoints
├── data/             # Dataset
├── data_source/      # Inputs for inference
├── image_test/       
├── assets/           # Images and video assets
├── train.py          # Training script
├── infer.py          # Inference script
└── requirements.txt
```

### 🚀 Usage
#### 🔧 Training
  ```bash
  python train.py
  ```

#### 🔍 Inference
```bash
1. Place input images into the 'data_source/' folder
2. Download the pretrained weights from:
    https://drive.google.com/drive/u/0/folders/1o_JQ31oXJ-QBaAgnmXquhMkuw7F6HWDk
 3. Run:
python infer.py
```

## 📂 Dataset
 **WiSARD Dataset:** https://sites.google.com/uw.edu/wisard/



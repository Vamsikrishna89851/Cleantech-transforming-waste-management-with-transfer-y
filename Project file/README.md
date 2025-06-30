# Waste Management Transfer Learning Project

## Overview
This project demonstrates using transfer learning to classify waste images into categories (e.g., plastic, metal, paper, glass, organic).

## Project Structure
- `src/preprocess.py`: Data loading and preprocessing.
- `src/train.py`: Model training using transfer learning (TensorFlow Keras).
- `src/app.py`: Flask application for inference.
- `requirements.txt`: Python dependencies.

## Setup
```bash
git clone <this-repo>
cd waste_management_transfer_learning_project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
- **Train**: `python src/train.py --data_dir /path/to/dataset`
- **Run app**: `python src/app.py`
Team id:LTVIP2025TMID46538

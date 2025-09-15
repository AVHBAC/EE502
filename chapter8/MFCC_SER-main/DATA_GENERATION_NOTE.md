# 📋 Data Generation Note

## Large Files Not Included in Repository

The following files are not included in this repository due to GitHub's 100MB file size limit:

- `features.csv` (~117MB) - Original extracted features from all datasets
- `enhanced_features.csv` (~2MB) - Augmented features (generated during experiments)

## How to Generate Required Data Files

### Option 1: Run Original Notebook
1. Open `Speech Emotion Recognition.ipynb`
2. Run all cells to generate `features.csv`
3. This will create the original feature dataset

### Option 2: Use Existing Dataset Processing
If you have the datasets in the correct structure:

```bash
# This will generate features.csv automatically
python -c "
import pandas as pd
from Speech_Emotion_Recogntion import *  # Import notebook functions
# Run feature extraction process
"
```

### Option 3: Start with Data Augmentation
If you don't need the original `features.csv`, you can start directly with:

```bash
# This creates enhanced_features.csv from raw audio
python advanced_data_augmentation.py
```

## Dataset Requirements

Make sure your `dataset/` folder contains:
- RAVDESS: `dataset/ravdess-emotional-speech-audio/`
- CREMA-D: `dataset/cremad/`  
- TESS: `dataset/toronto-emotional-speech-set-tess/`
- SAVEE: `dataset/surrey-audiovisual-expressed-emotion-savee/`

The scripts will automatically generate the required CSV files from these audio datasets.

## File Sizes Reference
- `data_path.csv`: ~1MB (included in repo)
- `features.csv`: ~117MB (generate locally)
- `enhanced_features.csv`: ~2MB (generated during experiment)

For questions about data generation, please see the main README.md or open an issue.
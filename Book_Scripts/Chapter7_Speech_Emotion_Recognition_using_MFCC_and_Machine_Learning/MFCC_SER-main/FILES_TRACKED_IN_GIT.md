# Files Tracked in Git Repository

## Core Code Files (✅ Tracked)

### Python Scripts
- **baseline_comparison.py** - Traditional ML model comparison script
  - Compares Random Forest, SVM, KNN, Logistic Regression
  - Can use original or enhanced features

### Jupyter Notebooks
- **Speech_Emotion_Recognition_ML.ipynb** - Complete ML pipeline
  - Feature extraction from audio
  - Data augmentation
  - Model training and evaluation
  - Visualization and results

### Configuration Files
- **requirements.txt** - Python package dependencies
- **.gitignore** - Git exclusion rules
- **.gitattributes** - Git attributes configuration

## Documentation Files (✅ Tracked)

- **README.md** - Main project documentation
  - Overview, features, installation
  - Quick start guide
  - Usage examples
  - Performance metrics
  - **Now includes**: Explanation of why silence trimming is not used

- **EXPERIMENT_RESULTS.md** - Detailed experimental documentation
  - Complete preprocessing pipeline
  - Feature extraction methodology
  - Model architectures and hyperparameters
  - Training procedures
  - Results analysis
  - Comparisons and insights

## Files NOT Tracked (❌ Gitignored)

### Generated Data Files
- `features.csv` - Extracted audio features (~50MB)
- `data_path.csv` - Audio file paths
- `enhanced_features.csv` - Augmented features
- `*.pkl` - Trained models and preprocessors
- `dataset/` - Raw audio files (too large)

### Auxiliary Scripts
- `generate_synthetic_features.py` - Demo feature generation
- `run_experiment.py` - Automated experiment runner
- `pdf_analysis_and_visualization.py` - Analysis utilities
- `RUN_EXPERIMENT.sh` - Shell script
- `setup.py` - Package setup (not needed for direct use)

### Documentation (Excluded)
- `PROJECT_SUMMARY.md` - Conversion summary
- `EXECUTION_ORDER_GUIDE.md` - Execution guide
- `CONTRIBUTING.md` - Contribution guide
- `DATA_GENERATION_NOTE.md` - Data notes
- `LICENSE` - License file

### Analysis and Outputs
- `pdf_analysis_output/` - PDF analysis results
- `chapter_8_overleaf/` - LaTeX document
- `*.pdf` - Generated PDFs
- Backup notebooks (`*_BACKUP.ipynb`)

### System Files
- `.claude/` - IDE configuration
- `__pycache__/` - Python cache
- `*.pyc` - Compiled Python
- `.DS_Store` - macOS files

---

## Summary

**Total Files in Git**: ~5-6 files
- 1 Python script (baseline_comparison.py)
- 1 Jupyter notebook (Speech_Emotion_Recognition_ML.ipynb)
- 2 Documentation files (README.md, EXPERIMENT_RESULTS.md)
- 2 Configuration files (requirements.txt, .gitignore)

**Philosophy**: Keep only essential code and documentation in version control. Exclude:
- Generated data (can be reproduced)
- Large binary files (models, datasets)
- Auxiliary/utility scripts
- IDE-specific files
- Temporary files

This keeps the repository clean, lightweight, and focused on the core ML implementation.

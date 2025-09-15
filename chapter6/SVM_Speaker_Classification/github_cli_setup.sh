#!/bin/bash
# GitHub CLI Setup Script for SVM Speaker Classification Analysis
# Run this script to publish the repository to your GitHub account

echo "🚀 Setting up SVM Speaker Classification Analysis on GitHub..."

# Navigate to project directory
cd /home/lab2208/Documents/EE502/chapter6/SVM_Speaker_Classification

# Initialize git if not already done
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
fi

# Add all files
echo "📝 Adding all files..."
git add .

# Create commit if there are changes
if git diff --cached --quiet; then
    echo "✅ No changes to commit"
else
    echo "💾 Creating initial commit..."
    git commit -m "Initial commit: Complete SVM speaker classification analysis

- Performance comparison across 5, 10, and 20 speakers  
- Synthetic audio dataset with 400 total samples
- MFCC feature extraction with noise reduction
- Comprehensive visualization and reporting
- Reproducible results with validation scripts

Results: 100% → 60% → 42.5% accuracy as speakers increase"
fi

# Set main branch
git branch -M main

# Create repository on GitHub and push
echo "🌐 Creating GitHub repository and pushing..."
gh repo create SVM-Speaker-Classification-Analysis \
    --public \
    --description "Comprehensive SVM performance analysis for speaker identification across different dataset sizes (5, 10, 20 speakers). Includes synthetic audio generation, MFCC feature extraction, and detailed performance comparisons." \
    --push

echo "🎉 Repository published successfully!"
echo "📖 View your repository at: https://github.com/$(gh api user --jq .login)/SVM-Speaker-Classification-Analysis"
#!/bin/bash

# 🚀 Complete Model Improvement Experiment Runner
# This script runs the entire model improvement pipeline in the correct order

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if file exists and get its size
check_file() {
    if [ -f "$1" ]; then
        size=$(du -h "$1" | cut -f1)
        print_success "✓ $1 exists (${size})"
        return 0
    else
        print_warning "✗ $1 not found"
        return 1
    fi
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check available RAM
    total_ram=$(free -h | awk '/^Mem:/ {print $2}')
    print_status "Available RAM: $total_ram"
    
    # Check disk space
    disk_space=$(df -h . | awk 'NR==2 {print $4}')
    print_status "Available disk space: $disk_space"
    
    # Check Python version
    python_version=$(python --version 2>&1)
    print_status "Python version: $python_version"
    
    # Check if GPU is available
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        print_status "GPU detected: $gpu_info"
    else
        print_warning "No GPU detected - will use CPU only"
    fi
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing required dependencies..."
    
    pip install --quiet optuna optuna-integration[keras] plotly scikit-optimize
    
    if [ $? -eq 0 ]; then
        print_success "Dependencies installed successfully"
    else
        print_error "Failed to install dependencies"
        exit 1
    fi
}

# Function to verify data files
verify_data() {
    print_status "Verifying data files..."
    
    if check_file "data_path.csv" && check_file "features.csv"; then
        # Check dataset size
        original_samples=$(python -c "import pandas as pd; print(len(pd.read_csv('data_path.csv')))")
        features_shape=$(python -c "import pandas as pd; print(pd.read_csv('features.csv').shape)")
        print_status "Original dataset: $original_samples samples"
        print_status "Features shape: $features_shape"
        return 0
    else
        print_error "Required data files not found!"
        exit 1
    fi
}

# Main experiment runner
run_experiment() {
    echo "================================================================================================"
    echo "🚀 STARTING COMPLETE MODEL IMPROVEMENT EXPERIMENT"
    echo "================================================================================================"
    echo "Estimated total runtime: 12-16 hours"
    echo "Make sure you have stable power and internet connection!"
    echo ""
    
    # Create results directory
    mkdir -p results
    mkdir -p logs
    
    START_TIME=$(date +%s)
    
    # Phase 1: Prerequisites (5-10 minutes)
    print_status "PHASE 1: Prerequisites and Setup"
    echo "================================================================================================"
    
    check_requirements
    install_dependencies
    verify_data
    
    # Backup important files
    print_status "Creating backups..."
    cp features.csv features_backup.csv 2>/dev/null || true
    cp data_path.csv data_path_backup.csv 2>/dev/null || true
    print_success "Backups created"
    
    # Phase 1 Complete
    phase1_time=$(($(date +%s) - START_TIME))
    print_success "Phase 1 completed in ${phase1_time}s"
    echo ""
    
    # Phase 2: Enhanced Feature Creation (2-4 hours)
    print_status "PHASE 2: Enhanced Feature Creation (This will take 2-4 hours)"
    echo "================================================================================================"
    
    if [ ! -f "enhanced_features.csv" ]; then
        print_status "Starting data augmentation - DO NOT INTERRUPT THIS STEP!"
        python advanced_data_augmentation.py 2>&1 | tee logs/augmentation.log
        
        if [ $? -eq 0 ] && [ -f "enhanced_features.csv" ]; then
            enhanced_size=$(du -h enhanced_features.csv | cut -f1)
            print_success "✓ Enhanced features created (${enhanced_size})"
        else
            print_error "Enhanced feature creation failed!"
            exit 1
        fi
    else
        print_warning "Enhanced features already exist, skipping..."
    fi
    
    phase2_time=$(($(date +%s) - START_TIME - phase1_time))
    print_success "Phase 2 completed in ${phase2_time}s"
    echo ""
    
    # Phase 3: Hyperparameter Optimization (2-3 hours)
    print_status "PHASE 3: Hyperparameter Optimization (2-3 hours)"
    echo "================================================================================================"
    
    print_status "Starting Optuna hyperparameter optimization..."
    python optuna_hyperparameter_tuning.py 2>&1 | tee logs/optuna.log
    
    if [ $? -eq 0 ]; then
        print_success "✓ Hyperparameter optimization completed"
    else
        print_error "Hyperparameter optimization failed - check logs/optuna.log"
        # Don't exit - continue with other phases
    fi
    
    phase3_time=$(($(date +%s) - START_TIME - phase1_time - phase2_time))
    print_success "Phase 3 completed in ${phase3_time}s"
    echo ""
    
    # Phase 4: Advanced Architecture Training (4-6 hours)
    print_status "PHASE 4: Advanced Architecture Training (4-6 hours)"
    echo "================================================================================================"
    
    print_status "Training advanced model architectures..."
    python advanced_models.py 2>&1 | tee logs/advanced_models.log
    
    if [ $? -eq 0 ]; then
        print_success "✓ Advanced models trained successfully"
    else
        print_error "Advanced model training had issues - check logs/advanced_models.log"
    fi
    
    phase4_time=$(($(date +%s) - START_TIME - phase1_time - phase2_time - phase3_time))
    print_success "Phase 4 completed in ${phase4_time}s"
    echo ""
    
    # Phase 5: Advanced Training Techniques (2-3 hours)
    print_status "PHASE 5: Advanced Training Techniques (2-3 hours)"
    echo "================================================================================================"
    
    print_status "Applying advanced training techniques..."
    python advanced_training_techniques.py 2>&1 | tee logs/advanced_training.log
    
    if [ $? -eq 0 ]; then
        print_success "✓ Advanced training techniques applied"
    else
        print_error "Advanced training had issues - check logs/advanced_training.log"
    fi
    
    phase5_time=$(($(date +%s) - START_TIME - phase1_time - phase2_time - phase3_time - phase4_time))
    print_success "Phase 5 completed in ${phase5_time}s"
    echo ""
    
    # Phase 6: Comprehensive Evaluation (30 minutes)
    print_status "PHASE 6: Comprehensive Evaluation"
    echo "================================================================================================"
    
    print_status "Running comprehensive comparison..."
    python baseline_comparison.py 2>&1 | tee logs/comparison.log
    
    print_status "Generating final results report..."
    python final_results_summary.py 2>&1 | tee logs/final_results.log
    
    phase6_time=$(($(date +%s) - START_TIME - phase1_time - phase2_time - phase3_time - phase4_time - phase5_time))
    print_success "Phase 6 completed in ${phase6_time}s"
    echo ""
    
    # Final Summary
    TOTAL_TIME=$(($(date +%s) - START_TIME))
    HOURS=$((TOTAL_TIME / 3600))
    MINUTES=$(((TOTAL_TIME % 3600) / 60))
    
    echo "================================================================================================"
    echo "🎉 EXPERIMENT COMPLETED SUCCESSFULLY!"
    echo "================================================================================================"
    echo "Total runtime: ${HOURS}h ${MINUTES}m"
    echo ""
    echo "Phase Breakdown:"
    echo "  Phase 1 (Setup): ${phase1_time}s"
    echo "  Phase 2 (Augmentation): ${phase2_time}s"
    echo "  Phase 3 (Optimization): ${phase3_time}s"
    echo "  Phase 4 (Advanced Models): ${phase4_time}s"
    echo "  Phase 5 (Training Techniques): ${phase5_time}s"
    echo "  Phase 6 (Evaluation): ${phase6_time}s"
    echo ""
    
    # Check results
    print_status "Checking final results..."
    
    if check_file "enhanced_features.csv"; then
        enhanced_samples=$(python -c "import pandas as pd; print(len(pd.read_csv('enhanced_features.csv')))" 2>/dev/null || echo "unknown")
        print_success "Enhanced dataset: $enhanced_samples samples"
    fi
    
    model_count=$(ls -1 *.h5 2>/dev/null | wc -l)
    print_success "Trained models: $model_count files"
    
    if check_file "IMPROVEMENT_REPORT.txt"; then
        print_success "✓ Comprehensive report generated"
        echo ""
        echo "📊 QUICK RESULTS PREVIEW:"
        echo "================================================================================================"
        head -20 IMPROVEMENT_REPORT.txt
        echo "..."
        echo "📄 Full report available in: IMPROVEMENT_REPORT.txt"
    fi
    
    echo ""
    echo "🎯 NEXT STEPS:"
    echo "  1. Review IMPROVEMENT_REPORT.txt for detailed results"
    echo "  2. Check individual log files in logs/ directory"
    echo "  3. Test best models on new data"
    echo "  4. Consider ensemble methods for further improvement"
    echo ""
    echo "🎉 Congratulations! Your model improvement experiment is complete!"
    echo "================================================================================================"
}

# Check if running in interactive mode
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being run directly
    echo "This will start a complete model improvement experiment."
    echo "Estimated time: 12-16 hours"
    echo "Do you want to continue? (y/N)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        run_experiment
    else
        echo "Experiment cancelled."
        exit 0
    fi
fi
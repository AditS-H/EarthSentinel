# EarthSentinel: Landslide Detection using Deep Learning

EarthSentinel is an advanced deep learning project that utilizes Siamese Neural Networks to detect and monitor landslides using satellite imagery time series data. The system analyzes temporal changes in terrain to identify potential landslide events, focusing on the Himachal Pradesh region.

## 🌟 Features

- **Siamese Neural Network Architecture**: Employs a twin neural network approach to compare temporal satellite imagery
- **Multi-temporal Analysis**: Processes weekly satellite imagery stacks to detect terrain changes
- **Automated Patch Generation**: Creates and processes image patches for efficient training
- **Pre-trained Models**: Includes trained models for immediate inference
- **Geographic Focus**: Specialized for Himachal Pradesh region with potential for adaptation to other areas

## 🗂️ Project Structure

```
├── images/                      # Weekly satellite image stacks
│   └── HP_week[1-14]_stack.tif # Time series imagery
├── patch_chunks/               # Processed image patches
├── models/                    # Trained model checkpoints
│   ├── best_model_fast.pth
│   ├── best_model_full.pth
│   └── siamese_model.pth
├── create_chunks_patches.py   # Patch generation script
├── generating_pairs.py        # Training pair generation
├── quick_train.py            # Fast training script
├── Batch_train.py            # Full batch training implementation
└── Global_Landslide_Catalog_Export.csv  # Ground truth data
```

## 🛠️ Technical Details

### Data Processing
- Processes multi-band satellite imagery
- Generates paired patches for Siamese network training
- Creates efficient data chunks for memory management
- Utilizes numpy arrays for fast data handling

### Model Architecture
- Siamese Neural Network for comparative analysis
- Checkpoint system for training recovery
- Multiple training modes (quick and full batch)
- Embedding generation for feature comparison

### Dataset
- Weekly satellite imagery time series
- Global Landslide Catalog for ground truth
- Geospatial data for Himachal Pradesh region

## 📊 Training Process

1. **Data Preparation**
   - Run `create_chunks_patches.py` to generate image patches
   - Execute `generating_pairs.py` to create training pairs

2. **Model Training**
   - Quick training: Use `quick_train.py` for rapid prototyping
   - Full training: Use `Batch_train.py` for complete model training

3. **Model Checkpoints**
   - Regular checkpoints saved during training
   - Best performing models saved separately

## 🔍 Model Outputs

The system generates:
- Feature embeddings for comparison
- Binary classification results for landslide detection
- Training and test set evaluations

## 📈 Performance

Model checkpoints are saved at:
- Regular intervals (checkpoint_epoch[1-5].pth)
- Best performing iterations (best_model_fast.pth, best_model_full.pth)

## 🤝 Contributing

Contributions to improve the model's accuracy or extend its capabilities are welcome. Please ensure to:
1. Follow the existing code structure
2. Document any new features
3. Test thoroughly before submitting changes

## 📝 License

This project is part of a research initiative. Please contact the repository owner for usage permissions.

## 👥 Authors

- AditS-H

## 🙏 Acknowledgments

- Global Landslide Catalog for providing ground truth data
- Satellite imagery providers for the temporal data
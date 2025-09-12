# How to Add a New Crop to Khet Guard

This guide explains how to add support for a new crop in the Khet Guard system, including data collection, model training, and integration.

## Overview

Adding a new crop involves:
1. **Data Collection**: Gathering disease/pest images for the new crop
2. **Data Preparation**: Organizing and preprocessing the dataset
3. **Model Training**: Training disease/pest detection models
4. **Integration**: Adding the crop to the recommendation system
5. **Testing**: Validating the new crop functionality

## Step 1: Data Collection

### Required Data Types
- **Disease Images**: Photos of various diseases affecting the crop
- **Pest Images**: Photos of pests that attack the crop
- **Healthy Images**: Photos of healthy crop plants
- **Metadata**: Location, season, weather conditions, etc.

### Data Sources
- **Field Photos**: Take photos in actual farming conditions
- **Research Datasets**: Use existing agricultural datasets
- **Collaboration**: Partner with agricultural universities/research centers
- **Crowdsourcing**: Collect data from farmers using the app

### Data Quality Requirements
- **Resolution**: Minimum 224x224 pixels, preferably 512x512 or higher
- **Lighting**: Good lighting conditions, avoid shadows
- **Angles**: Multiple angles (top, side, close-up)
- **Conditions**: Various weather and growth stages
- **Quantity**: Minimum 100 images per class, preferably 500+

## Step 2: Data Preparation

### Directory Structure
```
ml/datasets/new_crop/
├── train/
│   ├── healthy/
│   ├── disease_1/
│   ├── disease_2/
│   └── pest_1/
├── val/
│   ├── healthy/
│   ├── disease_1/
│   ├── disease_2/
│   └── pest_1/
└── test/
    ├── healthy/
    ├── disease_1/
    ├── disease_2/
    └── pest_1/
```

### Data Preprocessing
1. **Resize Images**: Standardize to 224x224 or 380x380 pixels
2. **Quality Check**: Remove blurry, duplicate, or poor quality images
3. **Augmentation**: Apply data augmentation techniques
4. **Validation Split**: Ensure balanced distribution across train/val/test

### Create NPZ Dataset
```bash
cd ml
python prepare_datasets.py --input datasets/new_crop --output data/processed/new_crop.npz
```

## Step 3: Model Training

### Update Configuration
Edit `train_pytorch_fusion.py` to include the new crop:

```python
CONFIG = {
    "models": {
        "disease": {"num_classes": 38, "data_path": "data/processed/plantvillage_color.npz"},
        "pest": {"num_classes": 20, "data_path": "data/processed/pest_dataset.npz"},
        "cattle": {"num_classes": 41, "data_path": "data/processed/cattle_dataset.npz"},
        "new_crop": {"num_classes": X, "data_path": "data/processed/new_crop.npz"}  # Add this
    },
    # ... rest of config
}
```

### Train the Model
```bash
python train_pytorch_fusion.py
```

### Validate Training
- Check TensorBoard logs: `tensorboard --logdir=model/exports/logs_new_crop`
- Verify model performance on validation set
- Test on unseen test data

## Step 4: Integration

### Update Labels
Add crop-specific labels to `ml/artifacts/new_crop/labels.json`:

```json
{
  "0": "healthy",
  "1": "disease_1_name",
  "2": "disease_2_name",
  "3": "pest_1_name"
}
```

### Update Pesticide Mapping
Add pesticide recommendations to `ml/recommender/pesticide_map.json`:

```json
{
  "new_crop_disease_1": {
    "recommended": ["Pesticide A", "Pesticide B"],
    "dosage": "2g per litre",
    "safety": "Wear protective equipment",
    "organic_alternatives": ["Neem oil", "Garlic extract"]
  }
}
```

### Update FastAPI Inference
The inference API will automatically detect the new model if it follows the naming convention.

## Step 5: Mobile App Integration

### Update Crop Selection
Add the new crop to the mobile app's crop selection screen:

```typescript
// In apps/mobile/src/screens/CropRecommendation.tsx
const CROPS = [
  { id: 'rice', name: 'Rice', icon: '🌾' },
  { id: 'wheat', name: 'Wheat', icon: '🌾' },
  { id: 'new_crop', name: 'New Crop', icon: '🌱' }, // Add this
];
```

### Update Disease/Pest Detection
The disease and pest detection screens will automatically work with the new crop once the model is deployed.

## Step 6: Testing

### Unit Tests
```bash
cd ml
python -m pytest tests/test_new_crop.py
```

### Integration Tests
1. **API Testing**: Test the inference API with new crop images
2. **Mobile Testing**: Test the mobile app with new crop functionality
3. **End-to-End Testing**: Complete workflow from image capture to recommendations

### Performance Validation
- **Accuracy**: Ensure model accuracy meets requirements (>85%)
- **Speed**: Verify inference time is acceptable (<2 seconds)
- **Robustness**: Test with various image conditions

## Step 7: Deployment

### Model Deployment
1. **Export Models**: Ensure ONNX models are generated
2. **Update API**: Deploy updated inference API
3. **Update Mobile**: Release new mobile app version

### Monitoring
- Set up monitoring for the new crop models
- Track accuracy and performance metrics
- Monitor user feedback and usage patterns

## Best Practices

### Data Collection
- **Diversity**: Collect data from multiple regions and seasons
- **Quality**: Prioritize image quality over quantity
- **Balance**: Ensure balanced representation of all classes
- **Validation**: Have agricultural experts validate the data

### Model Training
- **Hyperparameter Tuning**: Experiment with different learning rates and architectures
- **Cross-Validation**: Use k-fold cross-validation for robust evaluation
- **Ensemble Methods**: Consider ensemble models for better performance
- **Regular Updates**: Retrain models with new data periodically

### Integration
- **Backward Compatibility**: Ensure new features don't break existing functionality
- **User Experience**: Maintain consistent UI/UX across all crops
- **Documentation**: Keep documentation updated with new features
- **Support**: Provide adequate support for the new crop

## Troubleshooting

### Common Issues
1. **Low Accuracy**: Increase dataset size, improve data quality, or try different architectures
2. **Slow Inference**: Optimize model size, use quantization, or improve hardware
3. **Integration Issues**: Check API endpoints, model paths, and configuration files
4. **Mobile Issues**: Verify model compatibility and update app dependencies

### Getting Help
- Check the [API Documentation](API.md)
- Review the [Architecture Guide](ARCHITECTURE.md)
- Contact the development team
- Submit issues on the project repository

## Conclusion

Adding a new crop to Khet Guard is a comprehensive process that requires careful attention to data quality, model training, and system integration. Following this guide will help ensure a successful addition of new crop support to the platform.

Remember to:
- Start with high-quality data
- Thoroughly test all components
- Monitor performance after deployment
- Gather user feedback for continuous improvement

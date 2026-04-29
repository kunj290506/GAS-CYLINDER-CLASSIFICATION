# Gas Cylinder Detection System

An AI-powered solution to automatically detect and classify gas cylinders by brand, size, and color from images.

## Project Overview

This system uses a multi-task deep learning model based on transfer learning (EfficientNet/MobileNet) to predict:
- **Brand**: HP, Indane, or None
- **Size**: 2KG, 5KG, 15KG, or None
- **Color**: Blue, Red, or None
- **Presence**: Present or Absent

### Anti-Overfitting Features
- Heavy data augmentation for minority classes
- Class weighting to handle imbalanced data
- Progressive training with backbone freezing
- Dropout, batch normalization, and L2 regularization
- Early stopping and learning rate scheduling

## Project Structure

```text
D:/projects/gas/
├── src/
│   ├── config.py           # Configuration and hyperparameters
│   ├── prepare_data.py     # Data preparation and augmentation
│   ├── model.py            # Multi-task model architecture
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation and testing
├── streamlit_app/
│   ├── app.py              # Standalone Streamlit application
│   ├── best_model.pth      # Model weights for inference
│   └── requirements.txt    # Streamlit dependencies
├── backend/
│   ├── app.py              # FastAPI server
│   └── requirements.txt    # Backend dependencies
├── frontend/
│   ├── index.html          # Landing page
│   ├── upload.html         # Upload page
│   ├── styles.css          # Styling
│   └── script.js           # Frontend logic
├── data/
│   └── processed/          # Processed train/val/test data
├── models/
│   ├── best_model.pth      # Saved model weights
│   └── evaluation_results/ # Confusion matrices and metrics
├── logs/                   # TensorBoard logs
└── requirements.txt        # Main dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch with CUDA support (recommended) or CPU
- Node.js (for serving frontend) - optional

### Installation

**Install Main Python dependencies:**
```bash
pip install -r requirements.txt
```

## Data Preparation

Your data is located at `D:\PROJECT\GAS\DATA` with the following structure:
- **34,140 total images** across 8 classes
- Severe class imbalance (absent: 22 images vs others: 3,000-5,800)

**Prepare the data:**
```bash
python src/prepare_data.py
```

This will:
- Split data into 70% train, 15% validation, 15% test
- Apply heavy augmentation to minority classes
- Generate synthetic samples for "absent" class
- Save processed data to `data/processed/`
- Calculate and save class weights

## Training

The training process uses **progressive training** with 3 phases:

**Start training:**
```bash
python src/train.py
```

**Training phases:**
1. **Phase 1** (20 epochs): Freeze backbone, train classifier heads only
2. **Phase 2** (30 epochs): Unfreeze last 50% of backbone layers
3. **Phase 3** (20 epochs): Fine-tune entire model with lower learning rate

**Monitor training:**
```bash
tensorboard --logdir=logs
```

Open http://localhost:6006 to view training progress, loss curves, and accuracy metrics.

**Expected training time:**
- With GPU: 4-6 hours
- With CPU: 12-20 hours

## Evaluation

**Test the model on the test set:**
```bash
python src/evaluate.py
```

This generates:
- Confusion matrices for each task
- Per-class accuracy metrics
- Classification reports
- Results saved to `models/evaluation_results/`

### Evaluation Results

The multi-task model evaluation generates individual confusion matrices for Brand, Size, Color, and Presence classification tasks.

**Brand Classification**

![Confusion Matrix - Brand](models/evaluation_results/confusion_matrix_brand.png)

**Size Classification**

![Confusion Matrix - Size](models/evaluation_results/confusion_matrix_size.png)

**Color Classification**

![Confusion Matrix - Color](models/evaluation_results/confusion_matrix_color.png)

**Presence Classification**

![Confusion Matrix - Presence](models/evaluation_results/confusion_matrix_presence.png)

**Target metrics:**
- Overall accuracy: >85%
- Per-class F1-score: >0.75

## Running the Application

### Option 1: Standalone Streamlit App (Recommended)

A self-contained Streamlit application is available for easy deployment and testing.

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

### Option 2: FastAPI Backend + Static Frontend

**1. Start the Backend API**

```bash
cd backend
pip install -r requirements.txt
python app.py
```

The API will start on http://localhost:8000

**API Endpoints:**
- `GET /` - Health check
- `GET /api/info` - Model information
- `POST /predict` - Upload image for prediction
- Swagger docs: http://localhost:8000/docs

**2. Serve the Frontend**

Using Python's built-in server:
```bash
cd frontend
python -m http.server 3000
```
Access the application at http://localhost:3000

## Configuration

Edit `src/config.py` to customize:
- **Model parameters**: backbone, image size, dropout rate
- **Training parameters**: batch size, learning rate, epochs
- **Data augmentation**: rotation, brightness, zoom ranges
- **Class weights**: for handling imbalanced data

## Data Distribution

| Class | Image Count | Percentage |
|-------|-------------|------------|
| IND_5KG_BLUE_PRESENT | 5,828 | 17.1% |
| IND_15KG_RED_PRESENT | 5,654 | 16.6% |
| HP_5KG_BLUE_PRESENT | 5,362 | 15.7% |
| IND_5KG_RED_PRESENT | 5,130 | 15.0% |
| IND_2KG_BLUE_PRESENT | 4,802 | 14.1% |
| HP_15KG_RED_PRESENT | 3,868 | 11.3% |
| HP_2KG_BLUE_PRESENT | 3,474 | 10.2% |
| absent | 22 | 0.06% |

## Troubleshooting

**Model not found error**
- Make sure you have trained the model first: `python src/train.py`
- Check that `models/best_model.pth` exists

**CUDA out of memory**
- Reduce `BATCH_SIZE` in `src/config.py`
- Use a smaller backbone (mobilenet_v2 instead of efficientnet_b0)

**Backend connection error**
- Ensure the backend server is running on port 8000
- Check CORS settings in `backend/app.py`
- Update `API_URL` in `frontend/script.js` if using a different port

**Poor accuracy on new images**
- Collect more diverse training data
- Increase data augmentation strength
- Try different model architectures
- Ensure test images are similar quality to training images

## Next Steps

1. Collect more data for minority classes (especially "absent")
2. Test on real-world images from different environments
3. Deploy to production using Docker or cloud services
4. Add mobile app for on-device inference
5. Implement batch prediction for multiple images

## License

This project is created for gas cylinder detection and classification.

## Support

For issues or questions, please refer to the documentation or check the logs in:
- Training logs: `logs/`
- Evaluation results: `models/evaluation_results/`

---

**Built with PyTorch, Streamlit, FastAPI, and Modern Web Technologies**
# GAS-CYLINDER-CLASSIFICATION_PVT

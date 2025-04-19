# Brain Tumor Segmentation with MRI Scans

This project focuses on developing a machine learning model for segmenting brain tumors in MRI scans using the **LGG MRI Segmentation Dataset**. The goal is to accurately identify and delineate tumor regions, providing a foundation for assisting medical professionals in early diagnosis and treatment planning.

## 🚀 Project Overview
The project leverages deep learning to process MRI images and predict tumor regions. It includes data preprocessing, model training, and visualization of segmentation results. The model achieves robust performance, as demonstrated by real-time inference and evaluation metrics.

🎯 **Objective**: Segment tumor regions in brain MRI scans to support early diagnosis.

## 📊 Model Performance
- **Test Accuracy**: 91.01%
- **Dice Coefficient**: 0.88
- **Training Insights**: Consistent decrease in loss and improvement in validation accuracy across epochs, as visualized in training curves.

## 🧰 Tech Stack
- **Programming Language**: Python
- **Frameworks**: TensorFlow, Keras
- **Environment**: Google Colab
- **Visualization**: OpenCV, Matplotlib, Seaborn, Plotly
- **Data Handling**: Pandas, NumPy
- **Model Architecture**: ResNet50 (pre-trained) with custom classification head
- **Data Augmentation**: Keras ImageDataGenerator

## 📂 Project Structure
- **Dataset**: LGG MRI Segmentation dataset from Kaggle (`mateuszbuda/lgg-mri-segmentation`)
- **Code**: `medical_seg.py` (included in the repository)
  - Data preprocessing and loading
  - Model definition and training
  - Visualization of MRI scans, masks, and segmentation results
  - Evaluation metrics (accuracy, confusion matrix, classification report)
- **Outputs**:
  - Trained model weights: `clf-resnet-weights.keras`
  - Model architecture: `clf-resnet-model.json`
  - Visualizations: Plots for mask distribution, training curves, and segmentation overlays

## 🛠️ How to Run
1. **Prerequisites**:
   - Python 3.x
   - Install dependencies: `pip install -r requirements.txt`
   - Access to Google Colab or a local environment with GPU support (recommended)

2. **Dataset**:
   - Download the LGG MRI Segmentation dataset via Kaggle API:
     ```bash
     import kagglehub
     mateuszbuda_lgg_mri_segmentation_path = kagglehub.dataset_download('mateuszbuda/lgg-mri-segmentation')
     ```
   - Ensure the dataset is placed in the working directory or update paths in the script.

3. **Run the Code**:
   - Execute `medical_seg.py` in Google Colab or a local Jupyter environment.
   - The script handles:
     - Data loading and preprocessing
     - Model training with ResNet50
     - Visualization of results
     - Saving the model and weights

4. **View Results**:
   - Check generated plots for data distribution, training metrics, and segmentation visualizations.
   - Evaluate model performance using the test set metrics.

## 📈 Visualizations
- **Mask Distribution**: Bar plot showing the count of positive (tumor) and negative (no tumor) cases.
- **Training Curves**: Loss and accuracy plots for training and validation phases.
- **Segmentation Results**: Side-by-side visualizations of MRI scans, masks, and tumor overlays.

## 💡 Key Features
- **Data Preprocessing**: Sorting and pairing MRI images with corresponding masks, handling class imbalance.
- **Model Architecture**: Transfer learning with ResNet50, fine-tuned for binary classification (tumor vs. no tumor).
- **Training**: Early stopping, learning rate reduction, and model checkpointing to optimize performance.
- **Evaluation**: Comprehensive metrics including accuracy, confusion matrix, and classification report.

## 🌟 Future Improvements
- Experiment with advanced segmentation architectures (e.g., U-Net, Mask R-CNN).
- Incorporate multi-modal MRI data (e.g., FLAIR, T1, T2 sequences).
- Enhance model robustness with additional data augmentation techniques.
- Deploy the model as an API for real-time inference in clinical settings.

## 💬 Reflections
This project deepened my understanding of deep learning applications in healthcare, particularly in medical imaging. It highlights the potential of AI to support critical tasks like tumor detection, paving the way for impactful solutions in medicine.

## 🙌 Contributing
Feedback, suggestions, and collaborations are welcome! Feel free to:
- Open an issue for bugs or enhancements.
- Submit a pull request with improvements.
- Reach out for discussions on AI in healthcare.

## 🎓 Acknowledgments
- **Dataset**: [LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) by Mateusz Buda
- **Tools**: TensorFlow, Keras, Google Colab, Kaggle
- **Inspiration**: The potential of AI to transform healthcare

---

#MachineLearning #DeepLearning #AIInHealthcare #MedicalImaging #BrainTumorDetection #StudentProject #MRIsegmentation #AIForGood #OpenToWork #GoogleColab

# Brain-Stroke-Prediction-using-CNN

This project leverages a Convolutional Neural Network (CNN) to classify brain CT scan images into two categories: Normal and Stroke. The model is trained on a dataset of labeled images and achieves prediction performance using evaluation metrics like accuracy, ROC-AUC, PR-AUC, and classification reports. The final model can be used to predict new images interactively in a Colab environment.

**📂 Dataset**

    The dataset should be structured in a zip file named dataset.zip.

    It must contain two folders:

        /Normal: for CT images labeled as Normal.

        /Stroke: for CT images labeled as Stroke.

    The zip file is extracted in the Colab runtime from Google Drive.

**📦 Dependencies**

    Ensure the following Python libraries are installed (automatically available in Google Colab):

      tensorflow

      keras

      numpy

      matplotlib

      pandas

      PIL (Pillow)

      scikit-learn

      ipywidgets

**🚀 Project Workflow**

**1. Mount Google Drive**

    Mounts your Google Drive to access the dataset and save the model.

    from google.colab import drive
    drive.mount('/content/drive')

**2. Dataset Extraction**

    Unzips the dataset.zip file and reads the Normal and Stroke image data.

**3. Image Preprocessing**

    Images are resized to 224x224 and converted to RGB.

    Data is normalized (scaled between 0 and 1).

**4. Labeling**

    0 → Normal

    1 → Stroke

**5. Data Splitting**

    90% for training

    10% for testing

**6. CNN Model Architecture**

    3 Convolutional layers with ReLU activation and MaxPooling

    2 Dense layers with Dropout (20%) for regularization

    Final layer with sigmoid activation for binary classification

**7. Model Training**

    Epochs: 10

    Batch size: 32

    Optimizer: Adam

    Loss Function: Binary Crossentropy

**8. Model Evaluation**

    Includes:

    Accuracy and Loss (Test and Train)

    Confusion Matrix

    Classification Report

    ROC Curve and AUC

    Precision-Recall Curve

**9. Model Saving**

    Saved as cnn_model.keras to Google Drive for future use.

    📈 Model Metrics

**After training and evaluation:**

    Test and train accuracy are displayed.

    ROC-AUC and PR-AUC help validate model performance on imbalanced data.

**🖼️ Visualizations**

    Sample image grid with actual labels

    Test set predictions with image thumbnails

    Interactive prediction for uploaded images

**🧪 Predict New Images**

    Interactive image upload using ipywidgets in Google Colab:

        Upload a .jpg/.png image.

        Model predicts if the brain scan is Normal or Stroke.

**📌 Usage Notes**

      This notebook is built and tested in Google Colab.

      Upload your own brain scan images for testing.

      Use it for educational and research purposes. For clinical use, consult medical professionals.

**🔐 Author & Credits**

    Developed by: Krishna D , Srinath K

    Based on TensorFlow and Keras deep learning frameworks.

    Special thanks to open-source datasets and academic contributions.

**📞 Contact**

For queries or suggestions, feel free to reach out via [Email](mailto:krishnadayalan2005@gmail.com).

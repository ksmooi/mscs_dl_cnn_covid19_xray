# Classification of COVID-19 X-ray

## Introduction
This project focuses on classifying COVID-19 from X-ray images using Convolutional Neural Networks (CNN). Given the critical need for accurate and timely diagnosis during the pandemic, this project aims to leverage deep learning techniques to assist in the detection of COVID-19 from radiological images.

## Data Description

The dataset used in this project is designed to differentiate between normal and pneumonia-affected lungs, potentially aiding in the early detection of COVID-19 through radiological imaging. The COVID-19 X-ray Dataset is organized into training and testing sets, each containing images from two categories: NORMAL and PNEUMONIA.

### Folder Structure and File Details

The dataset is structured as follows:

```
data/test/NORMAL
data/test/PNEUMONIA
data/train/NORMAL
data/train/PNEUMONIA
```

### Summary of Files

| Folder           | Number of Files | File Format | Image Dimensions   |
|------------------|-----------------|-------------|--------------------|
| test/NORMAL      | 20              | JPEG        | (2244, 2030)       |
| test/PNEUMONIA   | 20              | JPEG        | (1294, 1022)       |
| train/NORMAL     | 74              | JPEG        | (1740, 1246)       |
| train/PNEUMONIA  | 74              | JPEG        | (882, 876)         |
| **Total Data Size** | **85.24 MB** |             |                    |

### Category-wise Summary

| Category        | Number of Samples | Mean Dimensions              | Dimension Standard Deviation        |
|-----------------|-------------------|------------------------------|-------------------------------------|
| Train Normal    | 74                | [1539.7, 1968.1, 3.0]        | [445.3, 337.95, 0.0]                |
| Train Pneumonia | 74                | [1231.3, 1427.0, 3.0]        | [842.95, 953.89, 0.0]               |
| Test Normal     | 20                | [1656.35, 2049.75, 3.0]      | [323.63, 237.64, 0.0]               |
| Test Pneumonia  | 20                | [1536.65, 1605.7, 3.0]       | [186.91, 229.14, 0.0]               |

### Key Points

- The dataset contains a total of 188 images, with 148 images in the training set (74 NORMAL and 74 PNEUMONIA) and 40 images in the test set (20 NORMAL and 20 PNEUMONIA).
- Images are in JPEG format with varying dimensions.
- The mean dimensions and standard deviations for each category indicate diversity in image sizes, necessitating resizing or normalization during preprocessing.
- The dataset size is 85.24 MB, making it manageable for typical machine learning and deep learning workflows.

## Exploratory Data Analysis (EDA)
EDA was performed to understand the dataset's size, dimensions, and structure. Sample images were visualized to observe the differences between normal and COVID-19 positive cases. The distribution of image dimensions and the class balance were also analyzed to ensure proper dataset handling during model training.

## Model Architecture
The CNN models were designed with varying configurations to identify the most effective architecture for COVID-19 classification. The key components of the architecture included:
- Convolutional layers with ReLU activation and max-pooling
- Fully connected layers with dropout for regularization
- Sigmoid activation for binary classification

Several models with different hyperparameters were tested, including variations in the number of layers, number of filters, dropout rates, and batch sizes.

## Results and Analysis
### Performance Metrics
The performance of each model was evaluated using metrics such as accuracy, precision, recall, and F1-score. The table below summarizes the final epoch metrics for each model:

| Model Name | Train Loss | Train Accuracy | Train Precision | Train Recall | Train F1 | Test Loss | Test Accuracy | Test Precision | Test Recall | Test F1 |
|------------|-------------|----------------|------------------|--------------|---------|-----------|---------------|----------------|-------------|--------|
| Model-A    | 0.083477    | 0.966216       | 0.972603         | 0.959459     | 0.965986| 0.048160  | 0.975         | 0.952381       | 1.000       | 0.975610|
| Model-B    | 0.214357    | 0.905405       | 0.884615         | 0.932432     | 0.907895| 0.060019  | 0.975         | 1.000000       | 0.950       | 0.974359|
| Model-C    | 0.115187    | 0.939189       | 0.922078         | 0.959459     | 0.940397| 0.048177  | 1.000         | 1.000000       | 1.000       | 1.000000|
| Model-D    | 0.048019    | 0.986486       | 0.973684         | 1.000000     | 0.986667| 0.013338  | 1.000         | 1.000000       | 1.000       | 1.000000|
| Model-E    | 0.112359    | 0.966216       | 0.972603         | 0.959459     | 0.965986| 0.045077  | 1.000         | 1.000000       | 1.000       | 1.000000|
| Model-F    | 0.130352    | 0.952703       | 0.985507         | 0.918919     | 0.951049| 0.068300  | 0.975         | 0.952381       | 1.000       | 0.975610|

### Performance Analysis
- **Number of Layers**: Model-A (3 layers) performed better than Model-B (5 layers), indicating that more layers did not necessarily improve performance.
- **Number of Filters**: Model-C, with more filters, showed perfect test metrics, suggesting enhanced feature extraction capabilities.
- **Dropout Rate**: Model-D and Model-E, with different dropout rates, both achieved perfect test metrics, highlighting the importance of dropout in preventing overfitting.
- **Batch Size**: Model-A (batch size 32) and Model-F (batch size 64) performed similarly, indicating that batch size did not significantly impact performance.

### Troubleshooting Steps
- Addressed tensor shape mismatches by correctly calculating the flattened feature map size after convolutional layers.
- Resolved tensor conversion issues by detaching tensors from the computation graph before converting them to NumPy arrays.
- Used dropout to mitigate overfitting and enhance model generalization.

### Hyperparameter Optimization Procedure
- Experimented with different numbers of layers, filters, dropout rates, and batch sizes to find the optimal model configuration.
- Found that a balanced architecture with appropriate dropout rates and filter sizes yielded the best performance while maintaining computational efficiency.

## Conclusion
### Summary of Results
- The project demonstrated the effectiveness of CNNs in classifying COVID-19 X-ray images.
- Models with balanced architectures and appropriate dropout rates showed the best performance.
- Hyperparameter tuning played a crucial role in achieving optimal model performance.

### Learning and Takeaways
- Gained insights into the importance of model architecture and hyperparameter tuning in CNNs.
- Understood the role of dropout rates in preventing overfitting and enhancing generalization.
- Recognized the significance of filter sizes in feature extraction.

### Analysis of Failures
- Additional layers in Model-B did not improve performance, likely due to overfitting or unnecessary complexity.
- Lower dropout rates or fewer filters led to poor generalization, emphasizing the need for balanced configurations.

### Suggestions for Improvement
- Experiment with different architectures like ResNet or DenseNet.
- Implement more sophisticated data augmentation techniques.
- Conduct extended hyperparameter tuning for learning rate, batch size, and optimizer types.
- Use a larger and more diverse dataset for better generalization.
- Consider ensemble methods to enhance performance and reliability.

## References
- [How to Build a Convolutional Neural Network (CNN) in PyTorch](https://www.youtube.com/watch?v=pDdP0TFzsoQ)
- [Data Augmentation in PyTorch](https://www.youtube.com/watch?v=HGwBXDKFk9I)
- [COVID-19 X-ray Dataset (Train & Test Sets) on Kaggle](https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets)
- [SMART-CT-SCAN_BASED-COVID19_VIRUS_DETECTOR](https://github.com/JordanMicahBennett/SMART-CT-SCAN_BASED-COVID19_VIRUS_DETECTOR/)
- [COVID-19 Chest X-ray Dataset by IEEE](https://github.com/ieee8023/covid-chestxray-dataset)

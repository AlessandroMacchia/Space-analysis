# Mission Status Classification Project

This project is focused on building a machine learning pipeline to classify the "Mission_Status" variable from a dataset sourced from Kaggle. The workflow covers data preprocessing, feature engineering, data visualization, and model building, resulting in a well-tuned classifier.

## Dataset
The dataset was obtained from Kaggle and includes information about various missions. Our target variable is `Mission_Status`, and the dataset contains both numerical and categorical features.

## Data Preprocessing and Feature Engineering
- **Handling Missing Values:** Imputation of missing data using the mean strategy.
- **Encoding Categorical Variables:** Using `LabelEncoder` for transforming categorical features.
- **Feature Engineering:** Starting from a single variable, we derived additional meaningful features to enhance model performance.
- **Balancing the Dataset:** Applied the SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.

## Data Visualization
We visualized key insights from the dataset, including:
- Organizations and Their Counts (ESA Highlighted)(img\Organizations and Their Counts (ESA Highlighted).png)
- Mission_Status (img\Mission_Status.png)
- Feature importance plots (img\FeatureImportance.png)
- Confusion matrix (img\confusionMatrix.png)
- ROC and Precision-Recall curves (img\ROC.png)

## Model
We used a Random Forest Classifier with hyperparameter tuning via Grid Search and Stratified Cross-Validation. The model’s performance was evaluated using:
- Confusion matrix
- Classification report
- ROC-AUC curve
- Precision-Recall curve




The results show strong performance for class 1 but a noticeable imbalance when it comes to class 0. Let’s break it down:

**RESULT: Classification Report:**
- **Class 0 (Minority Class):**
  - Precision: 0.54 — Only 54% of the predicted class 0 labels are actually class 0.
  - Recall: 0.49 — The model captures only 49% of the actual class 0 instances.
  - F1-score: 0.51 — A balance between precision and recall, but still quite low.
- **Class 1 (Majority Class):**
  - Precision: 0.94 — When the model predicts class 1, it’s correct 94% of the time.
  - Recall: 0.95 — The model captures 95% of the actual class 1 instances.
  - F1-score: 0.95 — A very strong performance on this class.
- **Overall Metrics:**
  - Accuracy: 90% — The model’s predictions are correct 90% of the time, but given the class imbalance, this metric alone doesn’t tell the whole story.
  - Weighted Average: Favors the performance of the dominant class, so the high overall metrics are mostly due to the model doing well on class 1.

**ROC Curve:**
- The ROC curve shows a good balance between true positive rate (sensitivity) and false positive rate. The area under the curve (AUC) is 0.85, indicating strong discriminatory ability. However, there’s still room for improvement, especially considering the imbalance in class performance.

**Precision-Recall Curve:**
- The Precision-Recall curve remains quite high, with precision close to 1 for a large range of recall values. This shows the model is generally good at minimizing false positives while capturing a high proportion of true positives. However, the sharp drop toward the end suggests the model struggles with the rare class when recall approaches 1.

**Overall Observation:**
The model performs well overall, but it clearly struggles with class 0 — likely due to the class imbalance despite applying SMOTE. It’s catching most of class 1, but precision and recall for the minority class remain low. Further tuning or experimenting with different algorithms, sampling strategies, or cost-sensitive methods might help address this imbalance.

## Acknowledgments
- Dataset: [Kaggle](https://www.kaggle.com/)



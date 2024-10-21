### **Study Overview**
Keratitis, an inflammatory condition of the cornea, is a significant cause of visual impairment in low- and middle-income countries (LMICs). The condition can be caused by bacteria, fungi, or amoeba, and timely, accurate diagnosis is essential for selecting the appropriate treatment and preserving vision. In LMICs, however, the limited availability and high cost of laboratory diagnostics often result in reliance on less accurate clinical observation. This study aimed to develop a deep learning framework to assist in diagnosing the source of keratitis infections using a Brazilian cornea dataset.

### **Dataset**
- **Source**: Brazilian cornea dataset
- **Total Observations**: 24,692
- **Positive Infection Cases**: 2,064
  - **Breakdown**:
    - Bacteria: 56.98%
    - Fungi: 13.42%
    - Amoeba: 10.03%
    - Bacteria and Fungi: 10.32%
    - Amoeba and Bacteria: 9.26%

### **Methodology**

#### **1. Biometric Feature Prediction**
This step aimed to assess whether biometric features like sex and age could be predicted from infected eye photographs. 

- **Sex Prediction**:
  - **Model**: DenseNet121 (pre-trained on ImageNet)
  - **Method**: 10-fold cross-validation for binary classification (sex prediction)
  - **Performance**: AUROC between 0.8790-0.8994 on the test set

- **Age Group Prediction**:
  - **Classes**: 
    - 0-18 years old
    - 18-40 years old
    - 40-65 years old
    - More than 65 years old
  - **Performance**: AUROC between 0.8331-0.8624 on the test set

#### **2. Disease Prediction**
Three approaches were implemented for infection type classification:

- **First Approach**: 
  - Three separate DenseNet models (pre-trained on ImageNet) were responsible for binary classification of each infection type.

- **Second Approach (Multitask V1)**:
  - A shared DenseNet backbone was used with three parallel classification layers, each responsible for one infection type (bacteria, fungi, amoeba).
  
- **Third Approach (Multitask V2)**:
  - A similar DenseNet backbone with a multi-head classification layer for multitask learning.

#### **3. Model Enhancements**
- **Clinical Loss Function**: Applied to the multitask V2 architecture.
- **Adaptive Threshold Methodology**: Youden’s Statistic was used to improve classification thresholds.

#### **Performance Comparison**
The performance was compared across the models, with Multitask V2 using clinical loss considered the best. Results across 10 folds for AUROC on the test set:

- **Bacteria**: 0.7413-0.7740
- **Fungi**: 0.8395-0.8725
- **Amoeba**: 0.9448-0.9616

### **Influence of Identity Features on Disease Prediction**
- **Statistical Analysis**:
  - **Sex**: T-test results indicated that sex significantly affects amoeba infection predictions.
  - **Age**: ANOVA results showed that age significantly affects fungi and bacteria infection predictions.

However, a more balanced dataset (age-wise) is recommended for more conclusive results.

### **Conclusion**
- The study successfully developed a deep learning framework that can accurately predict bacterial, fungal, and amoebic keratitis using corneal photographs.
- **Biometric Information**: Both sex and age were identifiable from the images, suggesting the potential benefit of using disentanglement techniques to ensure diagnosis models rely purely on medical features rather than demographic characteristics.







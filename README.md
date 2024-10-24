
## 1. **Context and Study Goals**
Keratitis is a significant cause of visual impairment, particularly in low- and middle-income countries (LMICs), where it accounts for around 10% of cases. This condition is often caused by bacterial, fungal, or amoebic infections. Due to limited access to laboratory diagnostics, clinicians in LMICs must often rely on clinical observation alone, which is less accurate. The goal of this study was to develop a deep learning framework capable of diagnosing the source of infection from corneal images, aiding more timely and accurate decision-making.

## 2. **Dataset Definition**
The study utilized a dataset of corneal images collected from a Brazilian cohort, comprising 24,692 observations. Of these, 2,064 were confirmed cases of infection, with the following distribution: 
- **56.98%** bacterial infections
- **13.42%** fungal infections
- **10.03%** amoeba infections
- **10.32%** bacterial and fungal co-infections
- **9.26%** amoeba and bacterial co-infections.

These images underwent preprocessing and were used to train and test the deep learning models.

## 3. **Patient Feature Prediction (Age and Sex)**
To evaluate how much biometric information (age and sex) could still be detected from infected corneal images, a DenseNet121 model pre-trained on ImageNet was used with a 10-fold cross-validation methodology:
- **Sex Prediction:** A binary classifier was trained to predict sex from corneal images, achieving an AUROC of **0.8790-0.8994** on the test set.
- **Age Prediction:** A multi-class classifier was trained to categorize age into four groups (0-18, 18-40, 40-65, and 65+ years). The AUROC ranged from **0.8331-0.8624** on the test set.

Several metrics, confusion matrices, and saliency maps were used for evaluation.

## 4. **Disease Prediction**
Three approaches were compared for predicting the type of infection:
1. **Single-task Approach:** Separate DenseNet architectures for each infection (bacteria, fungi, and amoeba).
2. **Multitask Approach V1:** A shared DenseNet backbone with parallel classification layers for each infection type.
3. **Multitask Approach V2:** A DenseNet backbone with a multi-head classification layer for multitask learning.

The best results were obtained with **Multitask V2** combined with a **Clinical Loss Function** and an **Adaptive Threshold Methodology (Youden’s Statistic)**. The AUROC values on the test set were:
- **Bacteria:** 0.7413-0.7740
- **Fungi:** 0.8395-0.8725
- **Amoeba:** 0.9448-0.9616

## 5. **Patient Feature Influence on Diagnosis**
To evaluate whether age and sex influence the infection predictions, statistical analyses were performed:
- **Sex Influence:** A T-test revealed that sex significantly affects the prediction of amoeba infections.
- **Age Influence:** ANOVA showed that age significantly affects the predictions of bacterial and fungal infections.

However, further analysis with a balanced age distribution is needed for stronger conclusions about the influence of age on the predictions.

## 6. **Conclusion**
This study demonstrated that it is possible to accurately predict the type of keratitis infection (bacterial, fungal, or amoebic) from corneal images using deep learning techniques. Additionally, age and sex information were found to be identifiable from the corneal photographs, suggesting the potential need for disentanglement strategies to ensure diagnosis focuses solely on medical features. These findings provide promising insights for improving diagnostic accuracy in LMICs where resources are limited.

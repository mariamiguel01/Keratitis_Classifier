Keratitis is an inflammatory corneal condition responsible for 10% of visual impairment in lowand
middle-income countries (LMICs); bacteria, fungi, or amoeba are the most common infection
etiology. An accurate and timely diagnosis of the infection agent is crucial not only for the treatment
option but also for the patient’s sight outcomes. Due to the high cost and limited availability
of laboratory diagnostics in LMICs, diagnosis is often made by clinical observation alone, despite
its lower accuracy. The goal of this study was to develop a deep learning framework model that
could assist in the diagnosis of the source of infection.
A Brazilian cornea dataset was utilized, comprising 24,692 observations, of which 2,064 had
positive results for infection (56.98% bacteria, 13.42% fungi, 10.03% for amoeba, 10.32% for
bacteria and fungi and lastly, 9.26% for amoeba and bacteria).
After preprocessing the dataset and preparing the images, different frameworks were implemented.
Firstly, the prediction of identity features was assessed to understand how much biometric
information the infected eye still had. To do so, a 10-fold methodology with a DenseNet121, pretrained
on ImageNet responsible for a binary decision regarding sex was implemented. The performance
was evaluated using several metrics, confusion matrices and saliency maps. An AUROC
of 0.8790-0.8994 was achieved on the test set. The same method was then repeated to classify
within age groups, with four classes being predicted (0-18 years old, 18-40 years old, 40-65 years
old and more than 65). An AUROC of 0.8331-0.8624 was achieved on the test set.
Regarding disease prediction, three different approaches were implemented and compared.
The first approach consisted of three DenseNet architectures (pre-trained on ImageNet), responsible
for a binary decision of each infection type. The second approach used a shared backbone
for the three tasks but three parallel classification layers, each responsible for a binary decision for
each infection, making it the V1 multitask approach. The third approach, multitask V2, consisted
of a similar DenseNet architecture used as backbone, with a multi-head classification layer for
multitask learning. Since multitask approach had interesting results, improvements were made to
that strategy. A Clinical Loss Function and an Adaptive Threshold Methodology (Youden’s Statistic)
were tested and assessed. The results were compared and Multitask V2 with clinical loss was
considered the best model, achieving a 95% confidence interval on the test set for AUROC, across
10 folds, of 0.7413-0.7740 for Bacteria, 0.8395-0.8725 for Fungi and 0.9448-0.9616 for Amoeba.
Finally, to assess the influence of identity features in this diagnosis, a statistic analysis was
performed using T-test, for sex comparison, and ANOVA, for age comparison. The results found
out that sex significantly affects the amoeba infection prediction and age significantly affects the
fungi and bacteria infection prediction; nevertheless, an analysis with a balanced dataset age-wise
should be implemented to have stronger conclusions about the age influence.
In conclusion, it was possible to correctly predict for BK,FK and AK when given an eye
corneal photograph. Furthermore, sex and age proved to be characteristics that are identifiable
from these photographs, which is why strategies like disentanglement could be beneficial to ensure
that only medical features are being used for the diagnosis.

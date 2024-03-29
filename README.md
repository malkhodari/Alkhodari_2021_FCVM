# Deep learning predicts preserved, mid-range, and reduced left ventricular ejection fraction from patients’ clinical profiles

This repository includes all codes needed to reproduce the work presented at "Alkhodari M, Jelinek H, Karlas A, Soulaidopoulos S, Arsenos P, Doundoulakis I, Gatzoulis K, Tsioufis K, Hadjileontiadis L, Khandoker A. Deep learning predicts preserved, mid-range, and reduced left ventricular ejection fraction from patients’ clinical profiles. Heart Failure and Transplantation, Frontiers in Cardiovascular Medicine. 2021. https://doi.org/10.3389/fcvm.2021.755968".

Estimating left ventricular ejection fraction (LVEF) levels and predicting LVEF-based heart failure (HF) groups using deep learning and patient clinical profiles.
HF groups were categorized based on the American Society of Echocardiography and the European Association of Cardiovascular Imaging (ASE/EACVI) guidelines [1-3] into HFpEF (EF > 55%), HFmEF (50% ≤ EF ≤ 55%), and HFrEF (EF < 50%).

This file includes the main code to perform two main tasks, namely estimations (regression) and prediction (classification), using pre-trained deep learning (DL), generalized linear model (GLM), and support vector machines (SVM) models. It includes a sample testing clinical information of an HFpEF patient. 

Paitnet clinical information includes: 1) Anti-arrhythmics medication, 2) Diuretic medication, 3) Age, 4) BMI, 5) Diabetes, 6) Ventricular tachycardia (VT), and 7) Prior myocardial infarction (Prior-MI).

Deep learning networks for regression and training are provided in deep_learning_networks.m

Make sure to cite it within your research!

1. Lang, R. M., Bierig, M., Devereux, R. B., Flachskampf, F. A., Foster, E., Pellikka, P. A., et al. (2006). Recommendations for chamber quantification. European journal of echocardiography, 7(2), 79-108.
2. Fonarow, G. C., Hsu, J. J. (2016). Left ventricular ejection fraction: what is “normal”?.‏ JACC: Heart Failure, 4(6). 511-513
3. Tsao, C. W., Lyass, A., Larson, M. G., Cheng, S., Lam, C. S., Aragam, J. R., et al. (2016). Prognosis of adults with borderline left ventricular ejection fraction. JACC: Heart Failure, 4(6), 502-510.

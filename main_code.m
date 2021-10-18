clc
clear all
close all

%% Loading Data
%%% No is 0
%%% Yes is 1
%%% BMI in Kg/m2

load testing_sample

testing_sample = table2array(testing_sample);
testing_sample_LVEF = testing_sample(8);
testing_sample_category = testing_sample(9);
testing_sample = testing_sample(1:7)';


%% Preparing array
%%% Adjusting categorical variables
%%% Sigmoid normalization using the original dataset used in the study

locations = find(testing_sample == 1); 
len_1 = 100;
all_numbers = 2:1/len_1:3;
for p = 1:length(locations)
    selected_location = locations(p,1);
    testing_sample(selected_location,1) = all_numbers(p+1);
end
    
locations = find(testing_sample == 0);
len_0 = 100;
all_numbers = 0:1/len_0:1;
for p = 1:length(locations)
    selected_location = locations(p,1);
    testing_sample(selected_location,1) = all_numbers(p);
end    

means = [0.579,1.253,57.987,27.277,0.784,0.639,1.972]';
stds = [0.492,1.014,10.437,3.471,0.760,0.590,0.931]';
testing_sample = testing_sample - means;
testing_sample = testing_sample ./ stds;
bottom = 1+exp((-testing_sample));
top = 1;
testing_sample = top./bottom;

%% Estimating LVEF (Regression)
%%% Using DL: Deep Learning Model
%%% Using GLM: Generalized Linear Model
%%% Using SVM: Support Vector Machines Model (RBF-kernel)

load Regression_DL_model
load Regression_GLM_model
load Regression_SVM_model

Etimated_LVEF_DL = predict(Regression_DL_model,testing_sample);
Etimated_LVEF_GLM = predict(Regression_GLM_model,testing_sample');
Etimated_LVEF_SVM = predict(Regression_SVM_model,testing_sample');

RMSE_DL = sqrt(mean((testing_sample_LVEF - Etimated_LVEF_DL).^2));
RMSE_GLM = sqrt(mean((testing_sample_LVEF - Etimated_LVEF_GLM).^2));
RMSE_SVM = sqrt(mean((testing_sample_LVEF - Etimated_LVEF_SVM).^2));

%% Predicting HF Category (Classification)
%%% Using DL: Deep Learning Model
%%% Using GLM: Generalized Linear Model
%%% Using SVM: Support Vector Machines Model (RBF-kernel)

load Classification_DL_model
load Classification_GLM_model
load Classification_SVM_model

[Predicted_HF_class_DL,scores_DL] = classify(Classification_DL_model,testing_sample);

[Predicted_HF_class_GLM,scores_GLM] = predict(Classification_GLM_model,testing_sample');
scores_GLM = 1 + scores_GLM;
scores_GLM(scores_GLM<0) = 0;
scores_GLM = scores_GLM./sum(scores_GLM);

[Predicted_HF_class_SVM,scores_SVM] = predict(Classification_SVM_model,testing_sample');
scores_SVM = 1 + scores_SVM;
scores_SVM(scores_SVM<0) = 0;
scores_SVM = scores_SVM./sum(scores_SVM);

Accuracy_DL = (sum(Predicted_HF_class_DL == categorical(testing_sample_category))./numel(categorical(testing_sample_category))).*100;
Accuracy_GLM = (sum(Predicted_HF_class_GLM == categorical(testing_sample_category))./numel(categorical(testing_sample_category))).*100;
Accuracy_SVM = (sum(Predicted_HF_class_SVM == categorical(testing_sample_category))./numel(categorical(testing_sample_category))).*100;



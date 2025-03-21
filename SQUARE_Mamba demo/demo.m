%========================================================================================================
% Input:
% Seven hydrometeorological variables from CRU dataset with fifteen history timesteps are the input data, 
% whose dimension is 15*7.
%--------------------------------------------------------------------------------------------------------
% Output:
% Prediction is the forecasting result of SQUARE-Mamba, whose dimension is 1*1.
%========================================================================================================
clear;clc;close all;
addpath(genpath('./functions'));
addpath(genpath('./main'));
%% Choose model
whichcase = 'mode 0: SQUARE-Mamba';
% whichcase = 'mode 1: SQUARE-Mamba w/o QLTEM';
switch whichcase
    case 'mode 0: SQUARE-Mamba'
        mode = 'SQUARE_Mamba';
    case 'mode 1: SQUARE-Mamba w/o QLTEM'
        mode = 'SQUARE_Mamba_wo_QLTEM';
end
%% Model testing
system(sprintf('python main/test_%s.py', mode));
gt = readmatrix(sprintf('./main/Result/%s/gt_Pooncarie.csv', mode));  
prediction = readmatrix(sprintf('./main/Result/%s/prediction_Pooncarie.csv', mode));
quantitative_index = assessment(gt, prediction);
%% Plot
curve_plot(gt, prediction, quantitative_index, mode);
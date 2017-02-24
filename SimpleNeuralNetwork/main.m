% Test function of Neural Network, for teaching purpose of WSE187.
% By: Xuewen Yang (xuewen.yang@stonybrook.edu)
% Created: 23-Feb-2017
% Last modified: 23-Feb-2017

%clear all;
%close all;
%clc;

%% train and test
%[train_labels, train_data, test_lables, test_data] = load_data();
train(train_labels,train_data,6000,20,0.01);
[accuracy, recognized_labels] = evaluate(test_data, test_lables);
disp(['Accuracy of model is: ' num2str(accuracy)]);

%% show the wrongly recognized data
i = 3;
figure;
imshow(test_data(:,:,i))
recognized_labels(i)
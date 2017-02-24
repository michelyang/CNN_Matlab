function [accuracy, recognized_labels] = evaluate(test_d, test_l)
    % accuracy: the correct_ones/all
    % data_l: the correct labels of data
    % wrong_labels: wrong labels of data
    load('model.mat');
%     A1 = zeros(h1, w1, n_kernel);  % feature maps
%     I2 = zeros(h2, w2);            
%     A2 = zeros(h2, w2, n_kernel);  % feature maps
%     % layer 3 - output layer
%     A3 = zeros(10, 1);
    wrong = 0;
    right = 0;
    num_test = size(test_d, 3);   % # of test data;
    data_l = zeros(1, num_test);  % the labels of the correct data
    wrong_labels = zeros(1, num_test);
    right_labels = zeros(1, num_test);
    recognized_labels = zeros(1, num_test);
    for i = 1 : num_test
        for j = 1 : n_kernel
            A1(:,:,j) = feedforward(test_d(:,:,i), W1(end:-1:1, end:-1:1, j), b1(j));
        end
        Z1 = sigmoid(A1);

        % layer 2: average/subsample with scaling and bias
        for  j = 1: n_kernel
            I2(:,:,j) = avgpool(Z1(:,:,j));
            A2(:,:,j) = I2(:,:,j) * S2(j) + b2(j);
        end
        Z2 = sigmoid(A2);
        % layer 3: fully connected
        for j = 1 : 10
            A3(j) = feedforward(Z2, W3(end:-1:1, end:-1:1, end:-1:1, j), b3(j));
        end
        Z3 = sigmoid(A3);

        [~, wrong_l] = max(Z3);  % wrong_l: mis-recognize as wrong_l 
        recognized_labels(i) = wrong_l - 1;
        if wrong_l ~= test_l(i) + 1;
            wrong = wrong + 1;
            data_l(wrong) = i;
            wrong_labels(wrong) = wrong_l - 1;
        else
            right = right + 1;
            right_labels(right) = wrong_l - 1;
        end
    end
    accuracy = right / num_test;
    disp(['Number of correctly recognizd numbers: ' num2str(right) ' out of ' num2str(num_test)]);
end
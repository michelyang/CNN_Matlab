function train(train_l, train_d, maxtrain, iter, eta)
    n_kernel = 10;  % # of kernels for layer 1
    s_kernel = 5;  % size of kernel of layer 1
    [h, w, n_train] = size(train_d); % height, width, and # of training data
    n_train = min(n_train, maxtrain);      % max # of training data used
    % initialization of network
    % layer 1
    h1 = h - s_kernel + 1;     % height and width after 1st layer of conv
    w1 = w - s_kernel + 1;
    A1 = zeros(h1, w1, n_kernel);  % feature maps

    % layer 2
    h2 = h1 / 2;                   % height and width after 2nd layer of pooling
    w2 = w1 / 2;
    I2 = zeros(h2, w2);            
    A2 = zeros(h2, w2, n_kernel);  % feature maps
    % layer 3 - output layer
    A3 = zeros(10, 1);

    % kernels for layer 1
    W1 = randn(s_kernel, s_kernel, n_kernel) * .01;
    b1 = ones(1, n_kernel);
    % scale parameters and bias for layer 2
    S2 = ones(1, n_kernel) * .01;
    b2 = ones(1, n_kernel);
    % weights and bias parameters for fully-connected output layer
    W3 = randn(h2, w2, n_kernel, 10) * .01;
    b3 = ones(10, 1);
    % true outputs
    Y = eye(10);

    % training with backprop
    for it = 1 : iter
        err = 0;
        for i = 1 : n_train   % number of data
            % ----------- forward propagation ------------ %
            % layer 1: convolution with bias followed by sigmoidal
            % squashing
            for c_k = 1 : n_kernel
                % the kernels should be flipped up side down, left
                % side right
                A1(:,:,c_k) = feedforward(train_d(:,:,i), W1(end:-1:1,end:-1:1,c_k), b1(c_k));
            end
            Z1 = sigmoid(A1);   % the feature map

            % layer2: average/subsampling with scaling and bias
            for c_k = 1 : n_kernel
                I2(:,:,c_k) = avgpool(Z1(:,:,c_k));  % do average pooling
                A2(:,:,c_k) = I2(:,:,c_k) * S2(c_k) + b2(c_k);
            end
            Z2 = sigmoid(A2);

            % layer 3: fully connected layer
            for j = 1 : 10
                A3(j) = feedforward(Z2, W3(end:-1:1, end:-1:1, end:-1:1, j), b3(j));
                %A3(j) = convn(Z2, W3(end:-1:1, end:-1:1, end:-1:1, j), ...
                    %'valid') + b3(j);
            end
            Z3 = sigmoid(A3);  % final output

            err = err + .5 * norm(Z3 - Y(:,train_l(i) + 1), 2)^2;

            % --------- backpropagation --------- %
            % compute error at output layer
            Del3 = (Z3 - Y(:,train_l(i) + 1)) .* sigmoid_prime(Z3);
            % compute error at layer 2
            Del2 = zeros(size(Z2));
            for j = 1 : 10
                Del2 = Del2 + Del3(j) * W3(:,:,:,j);
            end
            Del2 = Del2.*sigmoid_prime(Z2);
            % compute error at layer 1
            Del1 = zeros(size(Z1));
            for j = 1 : n_kernel
                Del1(:,:,j) = 4*S2(j).*sigmoid_prime(Z1(:,:,j));
                for k = 1 : h1
                    for iw = 1 : w1
                        Del1(k,iw,j) = Del1(k,iw,j) * ...
                            Del2(floor((k+1)/2), floor((iw+1)/2), j);
                    end
                end
            end
            % update bias at layer 3
            db3 = Del3;
            b3 = b3 - eta * db3;
            % update weights at layer 3
            for j = 1 : 10
                dW3 = db3(j) * Z2;
                W3(:,:,:,j) = W3(:,:,:,j) - eta * dW3;
            end
            % update scale and bias parameters at layer 2
            for j = 1 : n_kernel
                dS2 = convn(Del2(:,:,j), I2(end:-1:1, end:-1:1, j), 'valid');
                S2(j) = S2(j) - eta * dS2;
                db2 = sum(sum(Del2(:,:,j)));
                b2(j) = b2(j) - eta * db2;
            end
            % update kernel weights and bias parameters at layer 1
            for j = 1 : n_kernel
                dW1 = convn(train_d(:,:,i), Del1(end:-1:1, end:-1:1, j), 'valid');
                W1(:,:,j) = W1(:,:,j) - eta * dW1;
                db1 = sum(sum(Del1(:,:,j)));
                b1(j) = b1(j) - eta * db1;
            end
        end
        disp(['Error at ieration ' num2str(it) ' is: ' num2str(err)]);
    end
    % save model
    save('model.mat', 'W1', 'b1', 'S2', 'b2', 'W3', 'b3', 'n_kernel');
end



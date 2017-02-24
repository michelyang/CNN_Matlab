% sigmoid activation function
function activation = sigmoid(x)
    activation = 1.0 ./ (1.0 + exp(-x));
end

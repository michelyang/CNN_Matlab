% derivative of sigmoid function
function derivative = sigmoid_prime(x)
    derivative = sigmoid(x) .* (1 - sigmoid(x));
end
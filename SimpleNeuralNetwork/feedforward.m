% return the output of the network, x is input
function conv_res = feedforward(x, weight, b)
    conv_res = convn(x, weight, 'valid') + b;
end
% miscellaneous methods load data
function [train_labels, train_data, test_lables, test_data] = load_data()
    train_labels = loadMNISTLabels('train-labels.idx1-ubyte');
    train_data = loadMNISTImages('train-images.idx3-ubyte');
    test_lables = loadMNISTLabels('t10k-labels.idx1-ubyte');
    test_data = loadMNISTImages('t10k-images.idx3-ubyte');
end
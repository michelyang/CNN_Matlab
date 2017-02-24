% average pooling
function [img_pool] = avgpool(img)
    img_pool = zeros(size(img) / 2);
    for r = 1 : 2 : size(img, 1)
        for c = 1 : 2 : size(img, 2)
            img_pool((r+1)/2, (c+1)/2) = (img(r, c) + img(r+1, c) + ...
                img(r, c+1) + img(r+1, c+1)) / 4;
        end
    end
end
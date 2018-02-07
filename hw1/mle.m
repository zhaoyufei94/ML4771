function accuracy = mle(train_size, start_point, XX, Y)
% Compute accuracy of MLE Multiple Gaussian on given data
% train_size: number of data(out of 10000) used to train Multiple Gaussian MLE
% XX: preprocessed training & test data (10000 * 200)
% Y: label vector (10000 * 1)

n = zeros(10, 1);
PY = zeros(10,1);
mu = zeros(10, 200);
sigma = zeros(200, 200, 10);
s_1 = zeros(1, 10);
s_2 = zeros(200, 200, 10);
accuracy = -1;

for i = start_point :train_size + start_point - 1
    n(Y(i) + 1) = n(Y(i) + 1) + 1;
    mu(Y(i) + 1, :) = mu(Y(i) + 1, :) + XX(i, :);
end
for i = 1:10
    mu(i, :) = mu(i, :) / n(i);
    PY(i) = n(i) / train_size;
end

for i = start_point: train_size + start_point - 1
    sigma(:, :, Y(i) + 1) = sigma(:, :, Y(i) + 1) + (XX(i, :) ...
        - mu(Y(i) + 1, :))' * (XX(i, :) - mu(Y(i) + 1, :));
end

for i = 1: 10
    sigma(:, :, i) = sigma(:,:,i) / n(i) + 0.05 * eye(200);
    s_1(i) = det(sigma(:, :, i));
    s_2(:, :, i) = sigma(:, :, i)^-1;
end

correct = 0;
if start_point + train_size < 10000
for i = train_size + start_point: 10000
    max_P = -Inf;
    class = -1;
    for j = 1: 10
        %PXY = exp(-(XX(i, :) - mu(j, :)) * sigma(:, :, j)^-1 * ...
        %    (XX(i, :) - mu(j, :))'/2) / sqrt((2*pi)^200 * det(sigma(:, :, j)));
        logPXY = -(log(s_1(j)) + (XX(i,:) - mu(j,:)) * ...
            s_2(:, :, j) * (XX(i, :) - mu(j, :))') / 2;
        % arg max PXY * PY = arg max logPXY + log(PY)
        if logPXY + log(PY(j)) > max_P
            max_P = logPXY + log(PY(j));
            class = j - 1;
        end
    end
    if class < 0
        disp('error');
        return;
    end
    if class == Y(i)
        correct = correct + 1;
    end
end
end
if start_point > 1
for i = 1: start_point - 1
    max_P = -Inf;
    class = -1;
    for j = 1: 10
        %PXY = exp(-(XX(i, :) - mu(j, :)) * sigma(:, :, j)^-1 * ...
        %    (XX(i, :) - mu(j, :))'/2) / sqrt((2*pi)^200 * det(sigma(:, :, j)));
        logPXY = -(log(s_1(j)) + (XX(i,:) - mu(j,:)) * ...
            s_2(:, :, j) * (XX(i, :) - mu(j, :))') / 2;
        % arg max PXY * PY = arg max logPXY + log(PY)
        if logPXY + log(PY(j)) > max_P
            max_P = logPXY + log(PY(j));
            class = j - 1;
        end
    end
    if class < 0
        disp('error');
        return;
    end
    if class == Y(i)
        correct = correct + 1;
    end
end
end

accuracy = correct / (10000 - train_size);

disp(accuracy);

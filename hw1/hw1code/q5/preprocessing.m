f_var = zeros(1, 784); % variance of each feature
a = ones(10000, 1);

f_mean = mean(X); % mean of each feature

for i = 1: 10000
    f_var = f_var + (X(i, :) - f_mean).^2 / 10000;
end

% Choose top 200 features/columns with largest variance
[~, f] = sort(f_var, 'descend');
f = f(1, 1:200);
f = sort(f);

% Normalize each feature/column to standard gaussian distribution
XX = (X(:, f(1)) - f_mean(f(1)) * a) / sqrt(f_var(f(1)));
for i = 2:200
    XX = [XX, (X(:, f(i)) - f_mean(f(i)) * a) / sqrt(f_var(f(i)))];
end

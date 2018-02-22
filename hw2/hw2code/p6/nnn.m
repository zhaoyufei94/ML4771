function nnn(k, r, X, Y)
% 8, 0.5
    w1 = rand(1, k);
    w2 = rand(k, 1);
    b1 = rand(k, 1);
    b2 = rand(1, 1);
    n = size(X);
    n =n(1);
    for i = 1: n
        x = X(i);
        y = Y(i);
        % forward
        o1 = sigmoid(w1' * x + b1);  % k, 1
        o2 = sigmoid(w2' * o1 + b2); % 1, 1
        err = o2 - y; % 1, 1
        % update w2, b2
        delta = err * o2 * (1 - o2);  % 1, 1
        
        % update w1, b1
        for j = 1: k
            ddelta = delta * w2(j) * o1(j) * (1 - o1(j));
            w1(j) = w1(j) - r * ddelta * x;
            b1(j) = b1(j) - r * ddelta;
        end
        w2 = w2 - r * delta * o1;
        b2 = b2 - r * delta;
    end
    YY = zeros(2000, 1);
    for i = 1: n
        x = X(i);
        o1 = sigmoid(w1' * x + b1);
        YY(i) = sigmoid(w2' * o1 + b2);
    end
    figure;
    plot(X, [Y, YY]);
    xlabel("X");
    ylabel("Y");
    legend("Y", "Y'");
end
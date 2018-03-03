function [w1, w2, b1, b2] = nn(k, r, X, Y)
% the function to train and plot result of  single hidden layer neural network.
% k: number of neurons in hidden layer
% r: learn rate
% X, Y: training data points
% return value [w1, w2, b1, b2]: parameter of trained neural network.
    bar = waitbar(0);
    w1 = rand(1, k);
    w2 = rand(k, 1);
    b1 = rand(k, 1);
    b2 = rand(1, 1);
    % initialize the parameters.
    [n, ~] = size(X);
    loop = 0;
    while 1
        Err = 0;
        loop = loop + 1;
        delta_w1 = zeros(1, k);
        delta_w2 = zeros(k, 1);
        delta_b1 = zeros(k, 1);
        delta_b2 = 0;
        for i = 1: n
            x = X(i);
            y = Y(i);
            % feed forward
            o1 = sigmoid(w1' * x + b1);  % k, 1
            o2 = sigmoid(w2' * o1 + b2); % 1, 1
            err = o2 - y; % 1, 1
            Err = Err + abs(err);
            % compute residual for each layer
            delta = err * o2 * (1 - o2);  % 1, 1   
            delta_w2 = delta_w2 + delta * o1;
            delta_b2 = delta_b2 + delta;
            for j = 1: k
                ddelta = delta * w2(j) * o1(j) * (1 - o1(j));
                delta_w1 = delta_w1 + ddelta * x;
                delta_b1 = delta_b1 + ddelta;
            end  
        end
        delta_w1 = delta_w1 / n;
        delta_w2 = delta_w2 / n;
        delta_b1 = delta_b1 / n;
        delta_b2 = delta_b2 / n;
	% update the parameters
        w1 = w1 - r * delta_w1;
        w2 = w2 - r * delta_w2;
        b1 = b1 - r * delta_b1;
        b2 = b2 - r * delta_b2;
        if loop > 200000 || Err / n < 0.02
            disp(Err);
            disp(Err/n);
            break
        end
        m = ["training: ", num2str(loop)];
        waitbar(loop/200000, bar, m);
    end

    % compute the output of network for each x in X.
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

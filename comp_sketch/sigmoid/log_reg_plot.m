function [] = log_reg_plot()

    a = 5;
    b = 3.4;
    thetax = @(x) a * x + b;

    sigmoid = @(x) 1 ./ (1 + exp(-thetax(x)));
    d_sigmoid = @(x) sigmoid(x) .* (1 - sigmoid(x));
    d2_sigmoid = @(x) (d_sigmoid(x) - 2 .* sigmoid(x) .* d_sigmoid(x));
    
%     sigmoid2d = @(x, y) 1 ./ (1 + exp(-(1 + 2 * x + 3 * y)));
%     grad_x = @(x, y) (1.0 ./ (1 + exp(-(1 + 2 * x + 3 * y)))^2) * exp(-(1 + 2 * x + 3 * y)) * 2;
%     grad_y = @(x, y) (1.0 ./ (1 + exp(-(1 + 2 * x + 3 * y)))^2) * exp(-(1 + 2 * x + 3 * y)) * 3;
     
    range = -10:0.1:10;
    [argvalue, argmax] = max(d_sigmoid(range) .* a);
    x_max = range(argmax);
     
    figure;
%     axis equal;
    hold on;
    x = -10:0.1:10;
    y = sigmoid(x);
    plot(x, y, 'r-');
    y_d = d_sigmoid(x) .* a;
    plot(x, y_d, 'g-');
    plot(x_max, argvalue, 'm.', 'MarkerSize', 10);
    y_d2 = d2_sigmoid(x) .* a;
    plot(x, y_d2, 'b-');
    legend({'sigmoid', 'first-order derivative', 'maximum-point', 'second-order derivative'}, 'Location', 'northwest');
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    figure;
%     axis equal;
    hold on;
    x = -10:0.1:10;
    y = sigmoid(x);
    plot(x, y, 'r-');
    
    x = -10 + rand() * 20;
    xc = x;
    
    epsilon = 10^(-5);
    
    if x > range(argmax)
        s = -1;
    else
        s = +1;
    end
    
    plot(x, sigmoid(x), 'r.', 'MarkerSize', 15);
    n = 20;
    h = 1;
    for i=1:n
        slope = d_sigmoid(x);
        h = min(1, 0.01 * (argvalue / (slope + epsilon)));
        x = x + s * h * ((slope + epsilon) / abs(slope + epsilon));
        plot(x, sigmoid(x), '.', 'MarkerSize', 15);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    figure;
%     axis equal;
    hold on;
    x = -10:0.1:10;
    y = sigmoid(x);
    plot(x, y, 'r-');
    
    x = xc;
    
    plot(x, sigmoid(x), 'r.', 'MarkerSize', 15);
    n = 20;
    h = 1;
    for i=1:n
        slope = d_sigmoid(x);
        if slope <= 0.01
            h = 1;
        else
            h = 0.2;
        end
        x = x + s * h * ((slope + epsilon) / abs(slope + epsilon));
        plot(x, sigmoid(x), '.', 'MarkerSize', 15);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%     figure;
%     hold on;
%     x = -10:1:10;
%     y = -10:1:10;
%     [X, Y] = meshgrid(x, y);
%     Z = sigmoid2d(X, Y);
%     surf(X, Y, Z);
%     
%     x = -10 + rand() * 20;
%     y = -10 + rand() * 20;
%     
%     plot3(x, y, sigmoid2d(x, y), 'r.', 'MarkerSize', 15);
%     n = 10;
%     h = 1;
%     for i=1:n
%         gradient = [grad_x(x, y), grad_y(x, y)];
% %         x = x + h * grad_x(x, y) / norm(gradient);
% %         y = y + h * grad_y(x, y) / norm(gradient);
%         x = x + h * min(1, 1 / grad_x(x, y));
%         y = y + h * min(1, 1 / grad_y(x, y));
%         plot3(x, y, sigmoid2d(x, y), '.', 'MarkerSize', 15);
%     end
%     
end




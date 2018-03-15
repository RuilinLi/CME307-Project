function [X] = admm2(D_sq, hat_D_sq, anchor, N_x_adj, N_a_adj, beta, eps, true_x)
    max_it = 100;
    
    
end

% Compute the value of the objective function f(X, Y)
function [val] = objective_f(X, Y, D_sq, hat_D_sq, anchors, N_x_adj, N_a_adj)
    val = 0;

    n_verts = size(D_sq, 1);
    d = size(anchors, 1);

    % Iterate over all vertices
    for i = 1:n_verts
        i_range = (d * (i - 1) + 1):(d * i);
        xi = X(i_range);
        yi = Y(i_range);

        % Sum over adjacent vertices
        for j = N_x_adj{i}'
            j_range = (d * (j - 1) + 1):(d * j);
            xj = X(j_range);
            yj = Y(j_range);

            val = val + ((xi - xj)' * (yi - yj) - D_sq(j, i))^2;
        end

        % Sum over adjacent anchors
        for j = N_a_adj{i}'
            j_range = (d * (j - 1) + 1):(d * j);
            aj = anchors(:, j_range);

            val = val + ((aj - xi)' * (aj - yi) - hat_D_sq(j, i))^2;
        end
    end
end

% Compute the value of the langrangian L(X, Y, S) = f(X, Y) - S' (X - Y) +
% beta / 2 * || X - Y ||^2
function [val] = lagrangian(X, Y, S, D_sq, hat_D_sq, anchors, N_x_adj, N_a_adj, beta)
    val = 0;

    val = val + objective_f(X, Y, D_sq, hat_D_sq, anchors, N_x_adj, N_a_adj);

    val = val - S' * (X - Y) + beta / 2 * norm(X - Y)^2;
end

% Compute the gradient of f(X, Y) in X
function [grad] = gradient_X_f(X, Y, D_sq, hat_D_sq, anchors, N_x_adj, N_a_adj)
  
    n_verts = size(D_sq, 1);
    d = size(anchors, 1);
    grad = zeros(d * n_verts, 1);
   
    % Iterate over all vertices
    for i = 1:n_verts
        i_range = (d * (i - 1) + 1):(d * i);
        xi = X(i_range);
        yi = Y(i_range);
        
        % Sum over adjacent vertices
        for j = N_x_adj{i}'
            j_range = (d * (j - 1) + 1):(d * j);
            xj = X(j_range);
            yj = Y(j_range);
            
            grad(i_range) = grad(i_range) + ...
                2 * (yi - yj) * ((xi - xj)' * (yi - yj) - D_sq(j, i));
        end
        
        % Sum over adjacent anchors
        for j = N_a_adj{i}'
            j_range = (d * (j - 1) + 1):(d * j);
            aj = anchors(:, j_range);

            grad(i_range) = grad(i_range) - ...
                2 * xi * ((aj - xi)' * (aj - yi) - hat_D_sq(j, i));
        end
    end
end

function [grad] = gradient_X_lagrangian(X, Y, S, D_sq, hat_D_sq, anchors, ...
    N_x_adj, N_a_adj, beta)
    grad = gradient_X_f(X, Y, D_sq, hat_D_sq, anchors, N_x_adj, N_a_adj) - ...
        S' * X + beta * (X - Y);
end

function [grad] = gradient_Y_lagrangian(X, Y, D_sq, hat_D_sq, anchors, ...
    N_x_adj, N_a_adj, beta)
    grad = gradient_X_f(X, Y, D_sq, hat_D_sq, anchors, N_x_adj, N_a_adj) + ...
        S' * Y + beta * (X - Y);
end
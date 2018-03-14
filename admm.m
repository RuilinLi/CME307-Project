function [X] = admm(D_sq, hat_D_sq, anchor, N_x_adj, N_a_adj, beta, eps, true_x)
    d = size(anchor, 1);
    n = size(D_sq, 1);
    vec_size = d * n;

    Y = true_x(:);
    % Y = zeros(vec_size, 1);
    S = zeros(vec_size, 1);
    C = zeros(vec_size, 1);
    anchor_vec = anchor(:);
    
    max_it = 100;
    
    for it = 1:max_it
        Q = zeros(vec_size, vec_size);

        % Set up quadratic form
        for i = 1:n
           for j = 1:n
               i_range = (d*(i - 1) + 1):(d*i);
               j_range = (d*(j - 1) + 1):(d*j);

               if i ~= j
                    Q(i_range, j_range) = ...
                        -2 * (Y(i_range) - Y(j_range)) * (Y(i_range) - Y(j_range))';
               else
                   for k = N_x_adj{i}'
                      k_range = (d*(k - 1) + 1):(d*k);
                      Q(i_range, i_range) = Q(i_range, i_range) + ...
                          2 * (Y(i_range) - Y(k_range)) * (Y(i_range) - Y(k_range))';
                   end

                   for k = N_a_adj{i}'
                       k_range = (d*(k - 1) + 1):(d*k);
                       Q(i_range, i_range) = Q(i_range, i_range) + ...
                          2 * (Y(i_range) - anchor_vec(k_range)) * ...
                          (Y(i_range) - anchor_vec(k_range))';
                   end

                   Q(i_range, i_range) = Q(i_range, i_range) + beta * eye(d);
               end
           end
        end

        % Set up offset vector
        for i = 1:n
            i_range = (d*(i - 1) + 1):(d*i);

            for k = N_x_adj{i}'
                k_range = (d*(k - 1) + 1):(d*k);
                C(i_range) = C(i_range) - (2 * D_sq(i, k) * ...
                    (Y(i_range) - Y(k_range)));
            end

            for k = N_a_adj{i}'
                k_range = (d*(k - 1) + 1):(d*k);
                C(i_range) = C(i_range) - 2 * (Y(i_range) - anchor_vec(k_range)) * ...
                    (Y(i_range) - anchor_vec(k_range))' * anchor_vec(k_range);
                C(i_range) = C(i_range) - 2 * hat_D_sq(k, i) * ...
                    (Y(i_range) - anchor_vec(k_range));
            end

            flip = 2 * (1 - mod(it, 2)) - 1;
            C(i_range) = C(i_range) - flip * S(i_range) - beta * Y(i_range);
        end
        
        % Solve
        X = linsolve(Q, -C);
        
        % Update dual variables S after 2 iterations
        if mod(it, 2) == 0
            S = S - beta * (Y - X);
            
            val = abs(X' * Q * X / 2 + C' * X - S' * (Y - X) + ...
                    beta / 2 * norm(X - Y)^2);
            
            disp(val);
            if (val < eps)
                return;
            end
        end
        
        % Swap X and Y
        [Y, X] = deal(X, Y);
    end
end


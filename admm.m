function [X] = admm(D_sq, hat_D_sq, anchor, N_x_adj, N_a_adj, beta, eps)
    d = size(anchor, 1);
    n = size(D_sq, 1);
    vec_size = d * n;

    Y = zeros(vec_size, 1);
    S = zeros(vec_size, 1);
    C = zeros(vec_size, 1);
    anchor_vec = anchor(:);
    
    max_it = 100;
    
    for it = 1:max_it
        Q = zeroes(vec_size, vec_size);

        % Set up quadratic form
        for i = 1:n
           for j = 1:n
               i_range = (d*i - 1):(d*(i + 1));
               j_range = (d*j - 1):(d*(j + 1));

               if i ~= j
                    Q(i_range, j_range) = ...
                        -2 * (Y(i_range) - Y(j_range)) * (Y(i_range) - Y(j_range))';
               else
                   for k = N_x_adj{i}
                      k_range = (d*k - 1):(d*(k + 1));
                      Q(i_range, i_range) = Q(i_range, i_range) + ...
                          2 * (Y(i_range) - Y(k_range)) * (Y(i_range) - Y(k_range))';
                   end

                   for k = N_a_adj{i}
                       k_range = (d*k - 1):(d*(k + 1));
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
            i_range = (d*i - 1):(d*(i + 1)); 

            for k = N_x_adj{i}
                k_range = (d*k - 1):(d*(k + 1));
                C(i_range) = C(i_range) - 2 * D_sq(i, k) * ...
                    (Y(i_range) - Y(k_range))' * anchor_vec(k_range);
            end

            for k = N_a_adj{i}
                k_range = (d*k - 1):(d*(k + 1));
                C(i_range) = C(i_range) - 2 * (Y(i_range) - anchor_vec(k_range)) * ...
                    (Y(i_range) - anchor_vec(k_range))' * anchor_vec(k_range);
                C(i_range) = C(i_range) - 2 * hat_D_sq(i, k) * ...
                    (Y(i_range) - anchor_vec(k_range));
            end

            flip = 2 * (1 - mod(it, 2)) - 1;
            C(i_range) = C(i_range) - flip * S(i_range) - beta * Y(i_range);
        end
        
        % Solve
        X = linsolve(Q, c);
        
        % Update dual variables S after 2 iterations
        if mod(it, 2) == 0
            S = S - beta * (Y - X);
            
            if (abs(X' * Q * X / 2 - C' * X) - S' * (Y - X) + ...
                    beta / 2 * norm(X - Y)^2 < eps)
                return;
            end
        end
        
        % Swap X and Y
        [Y, X] = deal(X, Y);
    end
end


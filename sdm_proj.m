function [result, x] = sdm_proj(n, k, prolm)
    A = randn(n,n,k);
    b = randn(k,1);
    
    % Objective function
    A_dot_x = @(x) reshape(sum(sum(A.*x)),[],1);
    f = @(x) 1/2 * norm(A_dot_x(x) - b)^2;
    
    % Gradient of objective function
    dot_sum = @(B, v) sum(B.*permute(repmat(v,1,n,n),[3 2 1]), 3);
    grad_f = @(x) dot_sum(A,A_dot_x(x)) - dot_sum(A,b);
    
    % Initial guess
    x0 = rand(n,n);
    x0 = x0'*x0;
    
    switch prolm
        case 'LDL'
            [result, x] = graddesc_LDL(f, grad_f, x0, 1, 1E-5, n)
        case 'trunc'
            [result, x] = graddesc_trunc(f, grad_f, x0, 1, 1E-5, n)
    end
end

function [result, x] = graddesc_LDL(f, grad_f, x0, alpha0, eps, n)
  max_it = 10000;

  x = x0;
  result = f(x);
  fprintf('f(x) = %f\n', result);
 
  % Condition for step size
  % Define functions to compute the next x based on current x and direction and alpha
  next_x = @(a, y, d) y + a*d;
  size_cond = @(a, y, d) a <= (2 * (f(y) - f(y + a*d)) / norm(d(:))^2);

  for it=1:max_it
    fprintf('Iteration %i / %i\n', it, max_it);

    alpha = alpha0;
    dir = -grad_f(x);

	% Break if gradient is small
    if (norm(dir) < eps)
      break;
    end

    % Backward tracking step-size method
    while true      
      x_new = next_x(alpha, x, dir);
      % Use Cholesky decomposition to check if x_new is PD
      [R, p] = chol(x_new);
      if ((size_cond(alpha,x,dir)) & (p == 0))
        break;
      else
        alpha = alpha / 2;
      end
    end
    
    % Shift the negative diagonal entries of R to 0.
    proj_D = max(zeros(n,n), diag(diag(R)));
    % Build a matrix of 1's on diagonal
    ii = diag(true(n,1),0);
    % Set diagonal entries of R to 1
    R(ii) = 1; 
    % x_new = L*max{0,D}*L'
    x_new = R*proj_D*R';
    
    % Break if difference between positions is small
    if (norm(x_new - x) < eps)
      break;
    end

    x = x_new;
    result = f(x);
    fprintf('f(x_%i) = %i\n', it, result);
  end
end

function [result, x] = graddesc_trunc(f, grad_f, x0, alpha0, eps, n)
  max_it = 10000;

  x = x0;
  result = f(x);
  fprintf('f(x) = %f\n', result);
 
  % Condition for step size
  % Define functions to compute the next x based on current x and direction and alpha
  next_x = @(a, y, d) y + a*d;
  size_cond = @(a, y, d) a <= (2 * (f(y) - f(y + a*d)) / norm(d(:))^2);

  for it=1:max_it
    fprintf('Iteration %i / %i\n', it, max_it);

    alpha = alpha0;
    dir = -grad_f(x);

	% Break if gradient is small
    if (norm(dir) < eps)
      break;
    end

    % Backward tracking step-size method
    while true      
      x_new = next_x(alpha, x, dir);
      if (size_cond(alpha,x,dir))
        break;
      else
        alpha = alpha / 2;
      end
    end
    
    [V,D] = eigs(x_new, 6);
    d = [diag(D) ; zeros(n-6,1)];
    proj_d = max(zeros(n,1), d);
    
    x_new = zeros(n);
    for i = 1:6
        x_new = x_new + proj_d(i)*V(:,i)*V(:,i)';
    end
    
    % Break if difference between positions is small
    if (norm(x_new - x) < eps)
      break;
    end

    x = x_new;
    result = f(x);
    fprintf('f(x_%i) = %i\n', it, result);
  end
end
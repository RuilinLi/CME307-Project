function [result, x] = bb(f, grad_f, x0, alpha0, eps)
  % Some adaptation is applied to this function
  x_last = x0;
  x = x_last - alpha0 * grad_f(x_last);
  
  max_it = 10000;
  result = f(x);
  
  for it = 1:max_it
    fprintf('Iteration %i / %i\n', it, max_it);
    
    dir = grad_f(x);
    
    % Break if gradient is small
    if norm(dir,'fro') < eps
      break;
    end

    % Compute new position
    delta_x = x - x_last;
    delta_grad = grad_f(x) - grad_f(x_last);
    
    alpha = delta_x(:)' * delta_grad(:) / (delta_grad(:)' * delta_grad(:));
    x_new = x - alpha * grad_f(x);

    % Update position
    x_last = x;
    x = x_new;
    result = f(x);
    
     % Break if change in position is small enough
    if (norm(x - x_last) < eps)
      break;
    end
  end
end

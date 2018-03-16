function main(dimension,n_anchor,n_sensor,method, r, plot)
  rng(2018)
  % first generate anchors n_anchor recommended to be around 4
  anchor = 4*rand(dimension, n_anchor) - 2;

  % Now generate the sensors randomly, n_sensor recommended to be greater than 10
  sensor = 4*rand(dimension, n_sensor) - 2;

  % Now compute distances square
  inner_product_matrix = sensor'*sensor;
  sensor_norm = diag(inner_product_matrix);
  D_sq = sensor_norm*ones(1,n_sensor) + ones(n_sensor,1)*sensor_norm' - 2*inner_product_matrix;
  N_x = D_sq < r^2;
  N_x = N_x - diag(diag(N_x)); % Diagonals should be zero.
  anchor_norm = sum(anchor.^2,1)';
  hat_D_sq = anchor_norm*ones(1,n_sensor) + ones(n_anchor,1)*sensor_norm' - 2*anchor'*sensor;
  N_a = hat_D_sq < r^2;
  
  % Construct adjacency list between x's and x and a
  N_x_adj = cell(n_sensor,1); % the ith element of this cell is a list of sensors that are neighbours with i
  N_a_adj = cell(n_sensor,1); % the ith element of this cell is a list of anchors that are neighbours with i
  for i = 1:n_sensor
    N_x_adj{i} = find(N_x(:,i));
    N_a_adj{i} = find(N_a(:,i));
  end

  switch method
    case 'SOCP'
      cvx_begin
      variable X(dimension, n_sensor)
      minimize(1)
      subject to
      for i = 1:n_sensor
        neighbor_x = N_x_adj{i};
        neighbor_a = N_a_adj{i};
        for j = 1:size(neighbor_x,1)
          l = neighbor_x(j);
          norm(X(:,i) - X(:,l)) <= sqrt(D_sq(i,l));
        end
        for k = 1:size(neighbor_a,1)
          l = neighbor_a(k);
          norm(X(:,i) - anchor(:,l)) <= sqrt(hat_D_sq(l,i));
        end
      end
      cvx_end
      
    case 'SDP'
      cvx_begin sdp quiet
      variable Z(dimension+n_sensor,dimension+n_sensor) semidefinite
      minimize(1)
      subject to
      Z(1:dimension,1:dimension) == eye(dimension);
      for i = 1:n_sensor
        neighbor_x = N_x_adj{i};
        neighbor_a = N_a_adj{i};
        for j = 1:size(neighbor_x,1)
          l = neighbor_x(j);
          Z(dimension+i,dimension+i) + Z(dimension+l,dimension+l) - 2*Z(dimension+i,dimension+l) == D_sq(i,l);
        end
        for k = 1:size(neighbor_a,1)
          l = neighbor_a(k);
          indicator = zeros(n_sensor,1);
          indicator(i) = -1;
          vec = [anchor(:,l); indicator];
          quad_form(vec,Z) == hat_D_sq(l,i);
        end
      end
      Z >= 0;
      cvx_end
      X = Z(1:dimension,dimension+1:end);
      
    case 'LS'
      f = @(Y) compute_value(Y, anchor, D_sq, hat_D_sq, N_x_adj, N_a_adj);
      grad_f = @(Y) gradient(Y, anchor, D_sq, hat_D_sq, N_x_adj, N_a_adj);
      [value, X] = bb(f, grad_f, zeros(size(sensor)), 1, 1e-5, 10000);
      value
      
    case 'SDP-LS'
      cvx_begin sdp quiet
      variable Z(dimension+n_sensor,dimension+n_sensor) semidefinite
      minimize(1)
      subject to
      Z(1:dimension,1:dimension) == eye(dimension);
      for i = 1:n_sensor
        neighbor_x = N_x_adj{i};
        neighbor_a = N_a_adj{i};
        for j = 1:size(neighbor_x,1)
          l = neighbor_x(j);
          Z(dimension+i,dimension+i) + Z(dimension+l,dimension+l) - 2*Z(dimension+i,dimension+l) == D_sq(i,l);
        end
        for k = 1:size(neighbor_a,1)
          l = neighbor_a(k);
          indicator = zeros(n_sensor,1);
          indicator(i) = -1;
          vec = [anchor(:,l); indicator];
          quad_form(vec,Z) == hat_D_sq(l,i);
        end
      end
      Z >= 0;
      cvx_end
      
      X = Z(1:dimension,dimension+1:end);
      f = @(Y) compute_value(Y,anchor,D_sq,hat_D_sq, N_x_adj,N_a_adj);
      grad_f = @(Y) gradient(Y, anchor, D_sq, hat_D_sq, N_x_adj, N_a_adj);
      [value, X] = bb(f, grad_f,X , 1, 1e-5, 10000);
      disp(value)
      
    case 'ADMM'
          
      beta = 0.1;
      eps = 1E-2;
      X = admm2(D_sq, hat_D_sq, anchor, N_x_adj, N_a_adj, beta, eps, sensor);
      f = @(Y) compute_value(Y,anchor,D_sq,hat_D_sq, N_x_adj,N_a_adj);
      X = reshape(X, [dimension, size(sensor, 2)]);
      disp(f(X))
  end

  %sensor
  %Z(1:dimension,dimension+1:end)
  if plot
    color = [repmat([1,0,0],n_sensor,1); repmat([0,1,1],n_anchor,1); repmat([0,0,1],n_sensor,1)];
    scatter([sensor(1,:), anchor(1,:), X(1,:)], [sensor(2,:),anchor(2,:), X(2,:)],25,color,'filled');
    xlim([-4,4]);
    ylim([-4,4]);
  end
 %X
 %anchor
 %grad_block(X,anchor, D_sq,hat_D_sq,1,N_x_adj{1},N_a_adj{1})
 %compute_value(X,anchor,D_sq,hat_D_sq, N_x_adj,N_a_adj)
 %gradient(X, anchor, D_sq, hat_D_sq, N_x_adj, N_a_adj)
 %f = @(Y) compute_value(Y,anchor,D_sq,hat_D_sq, N_x_adj,N_a_adj);
 %grad_f = @(Y) gradient(Y, anchor, D_sq, hat_D_sq, N_x_adj, N_a_adj);
 %f(X)
 %grad_f(X)
end

% gradient of the least square problem in the ith variable
function [grad] = grad_block(X, A, D_sq, hat_D_sq,i,x_neighbor, a_neighbor)
% X is a matrix whose columns are current estiamte of sensor locations
% A is a matrix whose colulumns are the anchor locations
% the i,j entry is the square distance between x_i and x_j
% the k,i entry of hat_D_sq is the distance between a_k and x_i
% x_neighbor are the sensors neighbors of x_i, and a_neighbor are the anchor neighbors of x_i
  grad = zeros(size(X,1),1);
  for l = 1:size(x_neighbor,1)
    j = x_neighbor(l);
    grad = grad + 4*(norm(X(:,i) - X(:,j))^2 - D_sq(i,j))*(X(:,i) - X(:,j));
  end
  for l = 1:size(a_neighbor,1)
    k = a_neighbor(l);
    grad = grad + 4*(norm(A(:,k) - X(:,i))^2 - hat_D_sq(k,i))*(X(:,i) - A(:,k));
  end
end

function [grad] = gradient(X, A, D_sq, hat_D_sq, N_x_adj, N_a_adj)
  grad = zeros(size(X));
  for i = 1:size(X,2)
    x_neighbor = N_x_adj{i};
    a_neighbor = N_a_adj{i};
    grad(:,i) = grad_block(X, A, D_sq, hat_D_sq, i, x_neighbor, a_neighbor);
  end
end

function [value] = compute_value(X,A,D_sq,hat_D_sq, N_x_adj,N_a_adj)
  value = 0;
  for i = 1:size(X,2)
    x_neighbor = N_x_adj{i};
    a_neighbor = N_a_adj{i};
    for l = 1:size(x_neighbor,1)
      j = x_neighbor(l);
      value = value + (1/2)*(norm(X(:,i) - X(:,j))^2 - D_sq(i,j))^2;
    end
    for l = 1:size(a_neighbor,1)
      k = a_neighbor(l);
      value = value + (norm(X(:,i) - A(:,k))^2 - hat_D_sq(k,i))^2;
    end
  end
end

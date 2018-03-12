function main(dimension,n_anchor,n_sensor,method,plot)
	  % first generate anchors n_anchor recommended to be around 4
  anchor = 4*rand(dimension, n_anchor) - 2;

% Now generate the sensors randomly, n_sensor recommended to be greater than 10
  sensor = 4*rand(dimension, n_sensor) - 2;

				% Now compute distances square
  inner_product_matrix = sensor'*sensor;
  sensor_norm = diag(inner_product_matrix);
  D_sq = sensor_norm*ones(1,n_sensor) + ones(n_sensor,1)*sensor_norm' - 2*inner_product_matrix;
  anchor_norm = sum(anchor.^2,1)';
  hat_D_sq = anchor_norm*ones(1,n_sensor) + ones(n_anchor,1)*sensor_norm' - 2*anchor'*sensor;
  switch method
    case 'SOCP'
      cvx_begin
      variable X(dimension, n_sensor)
      minimize(1)
      subject to
      for i = 1:n_sensor
	for j = 1:(i-1)
	  norm(X(:,i) - X(:,j)) <= sqrt(D_sq(i,j));
	end
	for k = 1:n_anchor
	  norm(X(:,i) - anchor(:,k)) <= sqrt(hat_D_sq(k,i));
	end
      end
      cvx_end
    case 'SDP'
      cvx_begin sdp
      variable Z(dimension+n_sensor,dimension+n_sensor)
      minimize(1)
      subject to
      Z(1:dimension,1:dimension) == eye(dimension);
      for i = 1:n_sensor
	for j = 1:(i-1)
	  Z(dimension+i,dimension+i) + Z(dimension+j,dimension+j) - 2*Z(dimension+i,dimension+j) == D_sq(i,j);
	end
	for k = 1:n_anchor
	  indicator = zeros(n_sensor,1);
	  indicator(i) = -1;
	  vec = [anchor(:,k); indicator];
	  quad_form(vec,Z) == hat_D_sq(k,i);
	end
      end
      Z >= 0;
      cvx_end
  end

  sensor
  Z(1:dimension,dimension+1:end)
  if plot
    color = [ones(1,n_sensor), 1+ones(1,n_anchor), 2+ones(1,n_sensor)];
    scatter([sensor(1,:), anchor(1,:), X(1,:)], [sensor(2,:),anchor(2,:), X(2,:)],25,color,'filled');
  end
  
  
end

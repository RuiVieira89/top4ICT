function [w0] = func(X)
%FUNC Summary of this function goes here
%   Detailed explanation goes here


    try
		a = X(:,1);
		b = X(:,2);
		h = X(:,3);
    catch
		a = X(1);
		b = X(2);
		h = X(3);
    end


	x = a/2;
	y = b/2;

	s = b/a;
    E = 69e9;
    ne = 0.33;
    matDensity = 2500;
    
	alfa = 111e-6; % thermal diffusivity (m/s)
	Q0 = 3000*a*b; % point force (N)

	m = -1;
	n = -1;

	%% Rigidity matrix
	Dconst = (E*h.^3)/(12*(1-ne.^2));

	k = 0; % spring coeff (N/m)
	k_ = (k*b.^4)/(Dconst*pi.^4); 

	%% Thermal stresses
	T = 0; % Temperature (K)
	del_T = (T*alfa*Dconst*(1+ne)*pi.^2)/b.^2;

	qmn = 4*Q0/a*b;

	w0 = 0;
	w0_old = 0;
	cond = 0;
    
    while (cond == 0)

		m = m + 2;
		n = n + 2;

		Wmn = (b.^4)/(Dconst*pi.^4)*(qmn + del_T*(m.^2 * s.^2 + n.^2))/((m.^2 * s.^2 + n.^2).^2 + k_);

		w0 = w0 + Wmn*sin(m*pi*x/a)*sin(n*pi*y/b);
		
		cond = nanmax((w0 - w0_old)/w0) < 1e-8;
		w0_old = w0;
    end
    
    w0 = w0/(a*b*h*matDensity);
	%sigma_max = (6*qmn*2*b.^2)/(pi.^2*h.^2*(s.^2+1).^2)*(s.^2+ne); % maximum tension (Pa)

end


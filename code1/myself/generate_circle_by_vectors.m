function P_circle = generate_circle_by_vectors(t, C, r, n, u)
    n = n/norm(n);
    u = u/norm(u);
    
    nu = cross(n,u);

     xxyy = r*cos(t);
     yyxx = r*sin(t);

    % P_circle = r*cos(t)[:,newaxis]*u + r*sin(t)[:,newaxis]*nu+ C;
    P_circle = xxyy'*u + yyxx'*nu +C;
end
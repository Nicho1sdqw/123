function [P_rot] = rodrigues_rot(P, n0, n1)
    
    % # Get vector of rotation k and angle theta
    n0 = n0/norm(n0);
    n1 = n1/norm(n1);
    k = cross(n0,n1);
    k = k/norm(k);
    theta = acos(dot(n0,n1));
    N = size(P,1);
    % # Compute rotated points
    P_rot = zeros(N,3);
    for i = 1:size(P,1)
        P_rot(i,:) = P(i,:)*cos(theta) + cross(k,P(i,:))*sin(theta) + k*dot(k,P(i,:))*(1-cos(theta));
    end
    % return P_rot

end

clc;
for jj=1:10
% Find the posterior probablity estimate of each one
cov = Epsilon_hat_cell{jj}/(1); % Normalize the matrix with this factor to avoid compultational problems when you find determinant
cov = cov+(10^-11)*eye(size(cov));          % Make covariance matrix full rank
rank(cov)
%det(cov)
a=inv(cov);
end

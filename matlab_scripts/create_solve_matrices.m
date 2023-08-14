% Specific function test matrices

% Create 5x5 Toy Matrix
A_5_toy = [1, 2, 3, 4, 5;
           1, 0, 0, 0, 0;
           0, 1, 0, 0, 0;
           0, 0, 1, 0, 0;
           0, 0, 0, 1, 0];
b_5_toy = [3;2;4;2;2];
writematrix(A_5_toy, "solve_matrices\\A_5_toy.csv");
writematrix(b_5_toy, "solve_matrices\\b_5_toy.csv");

% Create 5x5 Matrix with solution can be found trivially
A_5_easysoln = [1, 0, 0, 0, 0;
                0, 2, 0, 0, 0;
                0, 0, 0, 0, 0;
                0, 0, 0, 0, 0;
                0, 0, 0, 0, 0];
b_5_easysoln = [1;2;0;0;0];
writematrix(A_5_easysoln, "solve_matrices\\A_5_easysoln.csv");
writematrix(b_5_easysoln, "solve_matrices\\b_5_easysoln.csv");

% Create backsubstitution test with matrices to test backsub in GMRES
% specifically
A_7_backsub = randn(7, 7);
b_7_backsub = randn(7, 1);
rho = norm(b_7_backsub);
[Q_7_backsub, R_7_backsub] = qr(A_7_backsub);
Q_8_backsub = [Q_7_backsub, zeros(7, 1);
               zeros(1, 7), 1];
R_8_backsub = [R_7_backsub;
               zeros(1, 7)];
writematrix(full(A_7_backsub), "solve_matrices\\A_7_dummy_backsub.csv");
writematrix(full(b_7_backsub), "solve_matrices\\b_7_dummy_backsub.csv");
writematrix(full(Q_8_backsub), "solve_matrices\\Q_8_backsub.csv");
writematrix(full(R_8_backsub), "solve_matrices\\R_8_backsub.csv");
rho_e1 = zeros(7, 1);
rho_e1(1) = rho;
x_1 = R_8_backsub(1:1, 1:1) \ (Q_8_backsub(1:1, 1:1)'*rho_e1(1:1));
x_2 = R_8_backsub(1:2, 1:2) \ (Q_8_backsub(1:2, 1:2)'*rho_e1(1:2));
x_3 = R_8_backsub(1:3, 1:3) \ (Q_8_backsub(1:3, 1:3)'*rho_e1(1:3));
x_4 = R_8_backsub(1:4, 1:4) \ (Q_8_backsub(1:4, 1:4)'*rho_e1(1:4));
x_5 = R_8_backsub(1:5, 1:5) \ (Q_8_backsub(1:5, 1:5)'*rho_e1(1:5));
x_6 = R_8_backsub(1:6, 1:6) \ (Q_8_backsub(1:6, 1:6)'*rho_e1(1:6));
x_7 = R_8_backsub(1:7, 1:7) \ (Q_8_backsub(1:7, 1:7)'*rho_e1(1:7));
writematrix(x_1, "solve_matrices\\x_1_backsub.csv");
writematrix(x_2, "solve_matrices\\x_2_backsub.csv");
writematrix(x_3, "solve_matrices\\x_3_backsub.csv");
writematrix(x_4, "solve_matrices\\x_4_backsub.csv");
writematrix(x_5, "solve_matrices\\x_5_backsub.csv");
writematrix(x_6, "solve_matrices\\x_6_backsub.csv");
writematrix(x_7, "solve_matrices\\x_7_backsub.csv");

% End-to-end matrices
convergence_tolerance_double = 1e-10;

% Create 64x64 convection diffusion with rhs sin(x)cos(y)
[A_convdiff64, b_convdiff64] = generate_conv_diff_rhs_sinxcosy(3, 0.5, 0.5);
x_convdiff64 = gmres( ...
    A_convdiff64, b_convdiff64, ...
    [], convergence_tolerance_double, 64 ...
);
writematrix(full(A_convdiff64), "solve_matrices\\conv_diff_64_A.csv");
writematrix(full(b_convdiff64), "solve_matrices\\conv_diff_64_b.csv");
writematrix(full(x_convdiff64), "solve_matrices\\conv_diff_64_x.csv");

% Create 256x256 convection diffusion with rhs sin(x)cos(y)
[A_convdiff256, b_convdiff256] = generate_conv_diff_rhs_sinxcosy(4, 0.5, 0.5);
x_convdiff256 = gmres( ...
    A_convdiff256, b_convdiff256, ...
    [], convergence_tolerance_double, 256 ...
);
% h = 1/15;
% [X, Y] = meshgrid(0:h:1, 0:h:1);
% u = sin(pi*X).*cos(pi*Y);
% v = A_convdiff256*(reshape(u, [256, 1]));
% surf(reshape(v, [16, 16]));
% hold on;
% surf(reshape(b_convdiff256+0.1, [16, 16]));
writematrix(full(A_convdiff256), "solve_matrices\\conv_diff_256_A.csv");
writematrix(full(b_convdiff256), "solve_matrices\\conv_diff_256_b.csv");
writematrix(full(x_convdiff256), "solve_matrices\\conv_diff_256_x.csv");

% Create 1024x1024 convection diffusion with rhs sin(x)cos(y)
[A_convdiff1024, b_convdiff1024] = generate_conv_diff_rhs_sinxcosy(5, 0.5, 0.5);
x_convdiff1024 = gmres( ...
    A_convdiff1024, b_convdiff1024, [], convergence_tolerance_double, 1024 ...
);
writematrix(full(A_convdiff1024), "solve_matrices\\conv_diff_1024_A.csv");
writematrix(full(b_convdiff1024), "solve_matrices\\conv_diff_1024_b.csv");
writematrix(full(x_convdiff1024), "solve_matrices\\conv_diff_1024_x.csv");

% Create Matrix random which should converge slowest
A_20_rand = randn(20, 20);
xtrue_20_rand = randn(20, 1);
b_20_rand = A_20_rand*xtrue_20_rand;
x_20_rand = gmres( ...
    A_20_rand, b_20_rand, ...
    [], convergence_tolerance_double, 20 ...
);
writematrix(A_20_rand, "solve_matrices\\A_20_rand.csv");
writematrix(b_20_rand, "solve_matrices\\b_20_rand.csv");
writematrix(x_20_rand, "solve_matrices\\x_20_rand.csv");

% Create saddle Matrix which should converge in roughly 3 steps
A_temp = randn(20, 20);
A_saddle = A_temp'*A_temp;
B_saddle = randn(5, 20);
saddle = [A_saddle, B_saddle'; B_saddle, zeros(5, 5)];
pre_cond = [A_saddle, zeros(20, 5);
            zeros(5, 20), B_saddle*inv(A_saddle)*B_saddle'];
inv_pre_cond = inv(pre_cond);
xtrue_saddle = randn(25, 1);
b_saddle = saddle*xtrue_saddle;
A_3eigs = inv_pre_cond*saddle;
b_3eigs = inv_pre_cond*b_saddle;
x_3eigs = gmres( ...
    A_3eigs, b_3eigs, [], convergence_tolerance_double, 3 ...
);
x_saddle = gmres( ...
    saddle, b_saddle, [], convergence_tolerance_double, 3, pre_cond ...
);
writematrix(A_3eigs, "solve_matrices\\A_25_3eigs.csv");
writematrix(b_3eigs, "solve_matrices\\b_25_3eigs.csv");
writematrix(x_3eigs, "solve_matrices\\x_25_3eigs.csv");
writematrix(saddle, "solve_matrices\\A_25_saddle.csv");
writematrix(b_saddle, "solve_matrices\\b_25_saddle.csv");
writematrix(x_saddle, "solve_matrices\\x_25_saddle.csv");
writematrix(inv_pre_cond, "solve_matrices\\A_25_invprecond_saddle.csv");

% Create lower/upper triangular to check substitution solve
A_2_temp = randi(100, 90, 90);
U_tri_90 = triu(A_2_temp);
x_90 = randi(100, 90, 1);
Ub_90 = U_tri_90*x_90;
writematrix(U_tri_90, "solve_matrices\\U_tri_90.csv");
writematrix(x_90, "solve_matrices\\x_tri_90.csv");
writematrix(Ub_90, "solve_matrices\\Ub_tri_90.csv");
L_tri_90 = tril(A_2_temp);
Lb_90 = L_tri_90*x_90;
writematrix(L_tri_90, "solve_matrices\\L_tri_90.csv");
writematrix(Lb_90, "solve_matrices\\Lb_tri_90.csv");

% Create Matrix and Inverse to test inverse preconditioner
A_inv_test = randi(45, 45);
Ainv_inv_test = inv(A_inv_test);
b_inv_test = randn(45, 1);
writematrix(A_inv_test, "solve_matrices\\A_inv_45.csv");
writematrix(Ainv_inv_test, "solve_matrices\\Ainv_inv_45.csv");
writematrix(b_inv_test, "solve_matrices\\b_inv_45.csv");

% Create ILU and sparse ILU
ilu_A = 10*randn(8, 8);
[ilu_L, ilu_U] = ilu(sparse(ilu_A));
writematrix(ilu_A, "solve_matrices\\ilu_A.csv");
writematrix(full(ilu_L), "solve_matrices\\ilu_L.csv");
writematrix(full(ilu_U), "solve_matrices\\ilu_U.csv");

ilu_sparse_A = 8*randn(8, 8);
ilu_sparse_A(1, 2) = 0; ilu_sparse_A(1, 3) = 0; ilu_sparse_A(1, 6) = 0; ilu_sparse_A(1, 7) = 0;
ilu_sparse_A(2, 1) = 0; ilu_sparse_A(2, 5) = 0; ilu_sparse_A(2, 6) = 0; ilu_sparse_A(2, 8) = 0;
ilu_sparse_A(3, 2) = 0; ilu_sparse_A(3, 4) = 0; ilu_sparse_A(3, 5) = 0; ilu_sparse_A(3, 7) = 0;
ilu_sparse_A(4, 1) = 0; ilu_sparse_A(4, 2) = 0; ilu_sparse_A(4, 3) = 0; ilu_sparse_A(4, 5) = 0;
ilu_sparse_A(5, 2) = 0; ilu_sparse_A(5, 3) = 0; ilu_sparse_A(5, 6) = 0; ilu_sparse_A(5, 7) = 0;
ilu_sparse_A(6, 3) = 0; ilu_sparse_A(6, 4) = 0; ilu_sparse_A(6, 7) = 0; ilu_sparse_A(6, 8) = 0;
ilu_sparse_A(7, 3) = 0; ilu_sparse_A(7, 5) = 0; ilu_sparse_A(7, 6) = 0; ilu_sparse_A(7, 8) = 0;
ilu_sparse_A(8, 2) = 0; ilu_sparse_A(8, 4) = 0; ilu_sparse_A(8, 5) = 0; ilu_sparse_A(8, 6) = 0;
for i=1:8
    ilu_sparse_A(i, i) = ilu_sparse_A(i, i) + ilu_sparse_A(i, i)/abs(ilu_sparse_A(i, i))*16;
    ilu_sparse_A(i, :) = i/8*ilu_sparse_A(i, :);
end
[ilu_sparse_L, ilu_sparse_U] = ilu(sparse(ilu_sparse_A));
writematrix(ilu_sparse_A, "solve_matrices\\ilu_sparse_A.csv");
writematrix(full(ilu_sparse_L), "solve_matrices\\ilu_sparse_L.csv");
writematrix(full(ilu_sparse_U), "solve_matrices\\ilu_sparse_U.csv");
options.type="ilutp"; options.droptol=0.01;
[ilu_sparse_L_0_01, ilu_sparse_U_0_01] = ilu(sparse(ilu_sparse_A), options);
writematrix(full(ilu_sparse_L_0_01), "solve_matrices\\ilut_0_01_sparse_L.csv");
writematrix(full(ilu_sparse_U_0_01), "solve_matrices\\ilut_0_01_sparse_U.csv");
options.type="ilutp"; options.droptol=0.1;
[ilu_sparse_L_0_1, ilu_sparse_U_0_1] = ilu(sparse(ilu_sparse_A), options);
writematrix(full(ilu_sparse_L_0_1), "solve_matrices\\ilut_0_1_sparse_L.csv");
writematrix(full(ilu_sparse_U_0_1), "solve_matrices\\ilut_0_1_sparse_U.csv");
options.type="ilutp"; options.droptol=0.2;
[ilu_sparse_L_0_2, ilu_sparse_U_0_2] = ilu(sparse(ilu_sparse_A), options);
writematrix(full(ilu_sparse_L_0_2), "solve_matrices\\ilut_0_2_sparse_L.csv");
writematrix(full(ilu_sparse_U_0_2), "solve_matrices\\ilut_0_2_sparse_U.csv");

function [A, b] = generate_conv_diff_rhs_sinxcosy(k, sigma, tau)

    % Calc n and h
    n = 2^k;
    h = 1/(n-1);
    
    % Form matrix of FDM on convection-diffusion equation within a sparse A
    A = sparse(n^2, n^2);
    
    % Add 4s on main diagonal
    D = 4*ones(n^2, 1);
    D(1:n, 1) = 1;
    D(n^2-n+1:n^2, 1) = 1;
    A = spdiags(D, 0, A);
    
    % Add -1-tau*h/2 for -1 diagonal
    low_D = (-1-tau*h/2)*ones(n, 1);
    low_D(1, 1) = 0;
    low_D(n, 1) = -2;
    low_D = repmat(low_D, n-1, 1);
    low_D = [zeros(n-1, 1); low_D];
    low_D(n^2-n:n^2) = 0;
    A = spdiags(low_D, -1, A);
    
    % Add -1+tau*h/2 for +1 diagonal
    high_D = (-1+tau*h/2)*ones(n, 1);
    high_D(1, 1) = -2;
    high_D(n, 1) = 0;
    high_D = repmat(high_D, n-1, 1);
    high_D = [zeros(n+1, 1); high_D];
    high_D(n^2-n+1:n^2) = 0;
    A = spdiags(high_D, 1, A);
    
    % Add -1+sigma*h/2 for +n diagonal
    C = (-1+sigma*h/2)*ones(n^2, 1);
    C(1:2*n) = 0;
    A = spdiags(C, n, A);
    
    % Add -1-sigma*h/2 for -n diagonal
    E = (-1-sigma*h/2)*ones(n^2-n, 1);
    E(n^2-2*n+1:n^2) = 0;
    A = spdiags(E, -n, A);
    
    % Calculate h^2*f as b
    [X, Y] = meshgrid(0:h:1, 0:h:1);
    u = sin(pi*X).*cos(pi*Y);
    b = 2*pi*u;
    b = b + sigma*cos(pi*X).*cos(pi*Y);
    b = b - tau*sin(pi*X).*sin(pi*Y);
    b = (pi*h^2)*b;
    b(1:n, 1) = 0;
    b(1:n, n) = 0;
    b = reshape(b, [n^2, 1]);

end
% Specific function test matrices

% Create 5x5 Toy Matrix (Well-conditioned A for testing but that
% does not converge before 5 iterations)
A_5_toy = [1, 0.1, 0.1, 0.1, 0.1;
           0.1, 2, 0.2, 0.2, 0.2;
           0.2, 0.2, 3, 0.3, 0.3;
           0.3, 0.3, 0.3, 4, 0.4;
           0.4, 0.4, 0.4, 0.4, 5];
b_5_toy = [3;2;4;2;2];
writematrix(A_5_toy, "solve_matrices\\A_5_toy.csv");
writematrix(b_5_toy, "solve_matrices\\b_5_toy.csv");
fprintf("5x5 Toy Mat Condition Number A: %0.5g\n", cond(A_5_toy));

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
% specifically (Well-conditioned A for testing)
A_7_backsub = generate_well_cond_rand_matrix(7, 10);
b_7_backsub = A_7_backsub*randn(7, 1);
rho = norm(b_7_backsub);
[Q_7_backsub, R_7_backsub] = qr(A_7_backsub);
Q_8_backsub = [Q_7_backsub, zeros(7, 1);
               zeros(1, 7), 1];
R_8_backsub = [R_7_backsub;
               zeros(1, 7)];
writematrix(A_7_backsub, "solve_matrices\\A_7_dummy_backsub.csv");
writematrix(b_7_backsub, "solve_matrices\\b_7_dummy_backsub.csv");
writematrix(Q_8_backsub, "solve_matrices\\Q_8_backsub.csv");
writematrix(R_8_backsub, "solve_matrices\\R_8_backsub.csv");
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

fprintf( ...
    "GMRES QR Triag. Solve Test Condition Number A: %0.5g\n", ...
    cond(A_7_backsub) ...
);
fprintf( ...
    "GMRES QR Triag. Solve Test Condition Number R: %0.5g\n", ...
    cond(R_8_backsub(1:7, 1:7)) ...
);

% End-to-end matrices
convergence_tolerance_double = 1e-10;

% Create 64x64 convection diffusion with rhs sin(x)cos(y)
[A_convdiff64, b_convdiff64] = generate_conv_diff_rhs_sinxcosy( ...
    8, 0.5, 0.5 ...
);
x_convdiff64 = gmres( ...
    A_convdiff64, b_convdiff64, ...
    [], convergence_tolerance_double, 64 ...
);
writematrix(full(A_convdiff64), "solve_matrices\\conv_diff_64_A.csv");
writematrix( ...
    full(A_convdiff64*A_convdiff64), ...
    "solve_matrices\\conv_diff_64_Asqr.csv" ...
);
writematrix(full(b_convdiff64), "solve_matrices\\conv_diff_64_b.csv");
writematrix(full(x_convdiff64), "solve_matrices\\conv_diff_64_x.csv");

fprintf( ...
    "Conv-Diff 64x64 Condition Number A: %0.5g\n", ...
    condest(A_convdiff64) ...
);

% Create 256x256 convection diffusion with rhs sin(x)cos(y)
[A_convdiff256, b_convdiff256] = generate_conv_diff_rhs_sinxcosy( ...
    16, 0.5, 0.5 ...
);
x_convdiff256 = gmres( ...
    A_convdiff256, b_convdiff256, ...
    [], convergence_tolerance_double, 256 ...
);
writematrix(full(A_convdiff256), "solve_matrices\\conv_diff_256_A.csv");
writematrix( ...
    full(A_convdiff256*A_convdiff256), ...
    "solve_matrices\\conv_diff_256_Asqr.csv" ...
);
writematrix(full(b_convdiff256), "solve_matrices\\conv_diff_256_b.csv");
writematrix(full(x_convdiff256), "solve_matrices\\conv_diff_256_x.csv");

fprintf( ...
    "Conv-Diff 256x256 Condition Number A: %0.5g\n", ...
    condest(A_convdiff256) ...
);

% Create 1024x1024 convection diffusion with rhs sin(x)cos(y)
[A_convdiff1024, b_convdiff1024] = generate_conv_diff_rhs_sinxcosy( ...
    32, 0.5, 0.5 ...
);
x_convdiff1024 = gmres( ...
    A_convdiff1024, b_convdiff1024, ...
    [], convergence_tolerance_double, 1024 ...
);
writematrix(full(A_convdiff1024), "solve_matrices\\conv_diff_1024_A.csv");
writematrix( ...
    full(A_convdiff1024*A_convdiff1024), ...
    "solve_matrices\\conv_diff_1024_Asqr.csv" ...
);
writematrix(full(b_convdiff1024), "solve_matrices\\conv_diff_1024_b.csv");
writematrix(full(x_convdiff1024), "solve_matrices\\conv_diff_1024_x.csv");

fprintf( ...
    "Conv-Diff 1024x1024 Condition Number A: %0.5g\n", ...
    condest(A_convdiff1024) ...
);

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

fprintf("Slow 20x20 Condition Number A: %0.5g\n", cond(A_20_rand));

% Create saddle Matrix which should converge in roughly 3 steps
% ensure well-conditioning
A_saddle = diag(abs(randn(10, 1)+10));
B_saddle = randn(2, 10);
saddle = [A_saddle, B_saddle'; B_saddle, zeros(2, 2)];
pre_cond = [A_saddle, zeros(10, 2);
            zeros(2, 10), B_saddle*inv(A_saddle)*B_saddle'];
inv_pre_cond = inv(pre_cond);
xtrue_saddle = randn(12, 1);
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

fprintf("Saddle Condition Number A: %0.5g\n", cond(saddle));
fprintf("Precond Saddle Condition Number A: %0.5g\n", cond(A_3eigs));

% Create lower/upper triangular to check substitution solve
A_2_temp = generate_well_cond_rand_matrix(90, 15);
U_tri_90 = triu(A_2_temp);
x_90 = 2*(randi(2, 90, 1)-1.5);
Ub_90 = U_tri_90*x_90;
writematrix(U_tri_90, "solve_matrices\\U_tri_90.csv");
writematrix(x_90, "solve_matrices\\x_tri_90.csv");
writematrix(Ub_90, "solve_matrices\\Ub_tri_90.csv");
L_tri_90 = tril(A_2_temp);
Lb_90 = L_tri_90*x_90;
writematrix(L_tri_90, "solve_matrices\\L_tri_90.csv");
writematrix(Lb_90, "solve_matrices\\Lb_tri_90.csv");

fprintf("Triag. Solve Test Condition Number A: %0.5g\n", cond(A_2_temp));
fprintf("Triag. Solve Test Condition Number U: %0.5g\n", cond(U_tri_90));
fprintf("Triag. Solve Test Condition Number L: %0.5g\n", cond(L_tri_90));

% Create Matrix and Inverse to test inverse preconditioner
A_inv_test = generate_well_cond_rand_matrix(45, 10);
Asqr_inv_test = A_inv_test*A_inv_test;
Ainv_inv_test = inv(A_inv_test);
b_inv_test = randn(45, 1);
writematrix(A_inv_test, "solve_matrices\\A_inv_45.csv");
writematrix(Asqr_inv_test, "solve_matrices\\Asqr_inv_45.csv");
writematrix(Ainv_inv_test, "solve_matrices\\Ainv_inv_45.csv");
writematrix(b_inv_test, "solve_matrices\\b_inv_45.csv");

fprintf( ...
    "Precond Test Condition Number A: %0.5g\n", ...
    cond(A_inv_test) ...
);
fprintf( ...
    "Precond Test Condition Number A^2: %0.5g\n", ...
    cond(Asqr_inv_test) ...
);
fprintf( ...
    "Precond Test Condition Number A^(-1): %0.5g\n", ...
    cond(Ainv_inv_test) ...
);
fprintf( ...
    "Precond Test Condition Number A^(-1)*A: %0.5g\n", ...
    cond(Ainv_inv_test*A_inv_test) ...
);
fprintf( ...
    "Precond Test Condition Number A^(-1)*A*A*A^(-1): %0.5g\n", ...
    cond(Ainv_inv_test*A_inv_test*A_inv_test*Ainv_inv_test) ...
);

% Create ILU and sparse ILU and pivoted and non-pivoted version
ilu_n = 64;
ilu_A = generate_well_cond_rand_matrix(ilu_n, 10);
[ilu_L, ilu_U] = ilu(sparse(ilu_A));
writematrix(ilu_A, "solve_matrices\\ilu_A.csv");
writematrix(full(ilu_L), "solve_matrices\\ilu_L.csv");
writematrix(full(ilu_U), "solve_matrices\\ilu_U.csv");
options.type = "ilutp";
options.droptol = 0;
[ilu_L_pivot, ilu_U_pivot, ilu_P_pivot] = ilu(sparse(ilu_A), options);
writematrix(full(ilu_L_pivot), "solve_matrices\\ilu_L_pivot.csv");
writematrix(full(ilu_U_pivot), "solve_matrices\\ilu_U_pivot.csv");
writematrix(full(ilu_P_pivot), "solve_matrices\\ilu_P_pivot.csv");

fprintf("ILU Test Condition Number Dense A: %g\n", condest(ilu_A));
fprintf("ILU Test Condition Number Dense L: %g\n", condest(ilu_L));
fprintf("ILU Test Condition Number Dense U: %g\n", condest(ilu_U));
fprintf( ...
    "ILU Test Condition Number Dense Pivot L: %g\n", ...
    condest(ilu_L_pivot) ...
);
fprintf( ...
    "ILU Test Condition Number Dense Pivot U: %g\n", ...
    condest(ilu_U_pivot) ...
);

ilu_sparse_A = generate_well_cond_sparse_rand_matrix(ilu_n, 5);
[ilu_sparse_L, ilu_sparse_U] = ilu(sparse(ilu_sparse_A));
writematrix(full(ilu_sparse_A), "solve_matrices\\ilu_sparse_A.csv");
writematrix(full(ilu_sparse_L), "solve_matrices\\ilu_sparse_L.csv");
writematrix(full(ilu_sparse_U), "solve_matrices\\ilu_sparse_U.csv");
[ilu_sparse_L_pivot, ilu_sparse_U_pivot, ilu_sparse_P_pivot] = ilu(sparse(ilu_sparse_A), options);
writematrix(full(ilu_sparse_L_pivot), "solve_matrices\\ilu_sparse_L_pivot.csv");
writematrix(full(ilu_sparse_U_pivot), "solve_matrices\\ilu_sparse_U_pivot.csv");
writematrix(full(ilu_sparse_P_pivot), "solve_matrices\\ilu_sparse_P_pivot.csv");

options4.type = "ilutp"; options4.droptol = 1e-4;
[ilu_L_4, ilu_U_4] = ilu(ilu_sparse_A, options4);
fprintf("ILU error 4: %g\n", norm(full(ilu_L_4*ilu_U_4-ilu_sparse_A)));

options3.type = "ilutp"; options3.droptol = 1e-3;
[ilu_L_3, ilu_U_3] = ilu(ilu_sparse_A, options3);
fprintf("ILU error 3: %g\n", norm(full(ilu_L_3*ilu_U_3-ilu_sparse_A)));

options2.type = "ilutp"; options2.droptol = 1e-2;
[ilu_L_2, ilu_U_2] = ilu(ilu_sparse_A, options2);
fprintf("ILU error 2: %g\n", norm(full(ilu_L_2*ilu_U_2-ilu_sparse_A)));

options1.type = "ilutp"; options1.droptol = 1e-1;
[ilu_L_1, ilu_U_1] = ilu(ilu_sparse_A, options1);
fprintf("ILU error 1: %g\n", norm(full(ilu_L_1*ilu_U_1-ilu_sparse_A)));

options0.type = "ilutp"; options0.droptol = 1e0;
[ilu_L_0, ilu_U_0] = ilu(ilu_sparse_A, options0);
fprintf("ILU error 0: %g\n", norm(full(ilu_L_0*ilu_U_0-ilu_sparse_A)));

fprintf("ILU Test Condition Number Sparse A: %g\n", condest(ilu_sparse_A));
fprintf("ILU Test Condition Number Sparse L: %g\n", condest(ilu_sparse_L));
fprintf("ILU Test Condition Number Sparse U: %g\n", condest(ilu_sparse_U));
fprintf("ILU Test Condition Number Sparse Pivot L: %g\n", condest(ilu_sparse_L_pivot));
fprintf("ILU Test Condition Number Sparse Pivot U: %g\n", condest(ilu_sparse_U_pivot));

function A = generate_well_cond_rand_matrix(n, diag_coeff_offset)
    A = randn(n, n);
    A = A + diag(diag_coeff_offset*diag(A)./abs(diag(A)));
end

function A = generate_well_cond_sparse_rand_matrix(n, diag_coeff_offset)
    A = sprandn(n, n, sqrt(n)/n);
    % Ensure non-zero diag and add offset
    d = randn(n, 1);
    d = (d+diag_coeff_offset*d./abs(d));
    A = spdiags(d, 0, A);
end

function [A, b] = generate_conv_diff_rhs_sinxcosy(n, sigma, tau)

    % Calc h
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
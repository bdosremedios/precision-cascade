#ifndef GMRES_H
#define GMRES_H

#include "../IterativeSolve.h"

template <template <typename> typename M, typename T, typename W=T>
class GMRESSolve: public TypedIterativeSolve<M, T>
{
protected:

    using TypedIterativeSolve<M, T>::typed_lin_sys;
    using TypedIterativeSolve<M, T>::init_guess_typed;
    using TypedIterativeSolve<M, T>::typed_soln;
    using TypedIterativeSolve<M, T>::max_iter;

    std::shared_ptr<Preconditioner<M, W>> left_precond_ptr;
    std::shared_ptr<Preconditioner<M, W>> right_precond_ptr;

    MatrixDense<T> Q_kry_basis = MatrixDense<T>(NULL);
    MatrixDense<T> H = MatrixDense<T>(NULL);
    MatrixDense<T> Q_H = MatrixDense<T>(NULL);
    MatrixDense<T> R_H = MatrixDense<T>(NULL);
    MatrixVector<T> next_q = MatrixVector<T>(NULL);

    int kry_space_dim;
    int max_kry_space_dim;
    double basis_zero_tol;
    T rho;

    void check_compatibility() const {

        if (!left_precond_ptr->check_compatibility_left(typed_lin_sys.get_m())) {
            throw std::runtime_error("Left preconditioner is not compatible with linear system");
        }
        if (!right_precond_ptr->check_compatibility_right(typed_lin_sys.get_n())) {
            throw std::runtime_error("Right preconditioner is not compatible with linear system");
        }
    
    }

    void set_initial_space() {

        kry_space_dim = 0;

        // Pre-allocate all possible space needed to prevent memory
        // re-allocation
        Q_kry_basis = MatrixDense<T>::Zero(
            typed_lin_sys.get_A_typed().get_handle(),
            typed_lin_sys.get_m(),
            typed_lin_sys.get_m()
        );
        H = MatrixDense<T>::Zero(
            typed_lin_sys.get_A_typed().get_handle(),
            typed_lin_sys.get_m()+1,
            typed_lin_sys.get_m()
        );
        Q_H = MatrixDense<T>::Identity(
            typed_lin_sys.get_A_typed().get_handle(),
            typed_lin_sys.get_m()+1,
            typed_lin_sys.get_m()+1
        );
        R_H = MatrixDense<T>::Zero(
            typed_lin_sys.get_A_typed().get_handle(),
            typed_lin_sys.get_m()+1,
            typed_lin_sys.get_m()
        );

        // Set rho as initial residual norm
        MatrixVector<T> Minv_A_x0(
            (left_precond_ptr->action_inv_M(typed_lin_sys.get_A_typed()*init_guess_typed)).template cast<T>()
        );
        MatrixVector<T> Minv_b(
            (left_precond_ptr->action_inv_M(typed_lin_sys.get_b_typed())).template cast<T>()
        );
        MatrixVector<T> r_0(Minv_b - Minv_A_x0);
        rho = r_0.norm();

        // Initialize next vector q as initial residual and mark as terminated if lucky break
        next_q = r_0;
        if (static_cast<double>(next_q.norm()) <= basis_zero_tol) { this->terminated = true; }

    }

    void initializeGMRES() {
        max_kry_space_dim = typed_lin_sys.get_m();
        if (max_iter > max_kry_space_dim) {
            throw std::runtime_error("GMRES outer iterations exceed matrix size");
        }
        check_compatibility();
        set_initial_space();
    }

    void update_subspace_k() {

        // Normalize next vector q and update subspace with it, assume that
        // checked in previous iteration that vector q was not zero vector
        int k = kry_space_dim-1;
        if (kry_space_dim == 0) {
            Q_kry_basis.get_col(kry_space_dim).set_from_vec(next_q/next_q.norm());
        } else {
            Q_kry_basis.get_col(kry_space_dim).set_from_vec(next_q/H.get_elem(k+1, k));
        }
        ++kry_space_dim;

    }

    void update_nextq_and_Hkplus1() {

        int k = kry_space_dim-1;

        // Find next vector power of linear system
        next_q = Q_kry_basis.get_col(k).copy_to_vec();
        next_q = (right_precond_ptr->action_inv_M(next_q)).template cast<T>(); // Apply action of right preconditioner
        next_q = typed_lin_sys.get_A_typed()*next_q; // Apply typed matrix A
        next_q = (left_precond_ptr->action_inv_M(next_q)).template cast<T>(); // Apply action of left preconditioner

        // Orthogonlize next_q to previous basis vectors and store coefficients/normalization in H
        T *h_vec = static_cast<T *>(malloc((typed_lin_sys.get_m()+1)*sizeof(T)));

        for (int i=0; i<=k; ++i) {
            // MGS from newly orthog q used for orthogonalizing next vectors
            MatrixVector<T> q_i(Q_kry_basis.get_col(i).copy_to_vec());
            h_vec[i] = q_i.dot(next_q);
            next_q -= q_i*h_vec[i];
            // H.coeffRef(i, k) = q_i.dot(next_q);
            // next_q -= q_i*H.coeff(i, k);
        }
        h_vec[k+1] = next_q.norm();
        for (int i=k+2; i<(typed_lin_sys.get_m()+1); ++i) {
            h_vec[i] = static_cast<T>(0);
        }
        H.get_col(k).set_from_vec(
            MatrixVector<T>(typed_lin_sys.get_A().get_handle(), h_vec, typed_lin_sys.get_m()+1)
        );
        // H.coeffRef(k+1, k) = next_q.norm();

        free(h_vec);

    }

    void update_QR_fact() {

        // Initiate next column of QR fact as most recent of H
        int k = kry_space_dim-1;
        R_H.get_col(k).set_from_vec(H.get_col(k).copy_to_vec());

        // Apply previous Given's rotations to new column
        MatrixDense<T> Q_H_block(Q_H.get_block(0, 0, k+1, k+1).copy_to_mat());
        R_H.get_block(0, k, k+1, 1).set_from_vec(
            Q_H_block.transpose_prod(H.get_col(k).copy_to_vec().slice(0, k+1))
        );

        // Apply the final Given's rotation manually making R_H upper triangular
        T alpha = R_H.get_elem(k, k);
        T beta = R_H.get_elem(k+1, k);
        T r_sqr = alpha*alpha + beta*beta; // Explicit intermediate variable to ensure
                                           // no auto casting into sqrt
        T r = std::sqrt(r_sqr);
        T c = alpha/r;
        T s = -beta/r;
        R_H.set_elem(k, k, r);
        R_H.set_elem(k+1, k, static_cast<T>(0));

        MatrixVector<T> Q_H_col_k(Q_H.get_col(k).copy_to_vec());
        MatrixVector<T> Q_H_col_kp1(Q_H.get_col(k+1).copy_to_vec());
        Q_H.get_col(k).set_from_vec(Q_H_col_k*c - Q_H_col_kp1*s);
        Q_H.get_col(k+1).set_from_vec(Q_H_col_k*s + Q_H_col_kp1*c);

    }

    void update_x_minimizing_res() {

        // Calculate RHS to solve
        MatrixVector<T> rho_e1(
            MatrixVector<T>::Zero(typed_lin_sys.get_A_typed().get_handle(), kry_space_dim+1)
        );
        rho_e1.set_elem(0, rho);
        MatrixVector<T> rhs(
            Q_H.get_block(0, 0, kry_space_dim+1, kry_space_dim+1).copy_to_mat().transpose_prod(rho_e1)
        );

        // Use back substitution to solve
        MatrixVector<T> y(
            R_H.get_block(0, 0, kry_space_dim, kry_space_dim).copy_to_mat().back_sub(
                rhs.slice(0, kry_space_dim)
            )
        );

        // Update typed_soln adjusting with right preconditioning
        MatrixDense<T> Q_kry_basis_block(
            Q_kry_basis.get_block(0, 0, typed_lin_sys.get_m(), kry_space_dim).copy_to_mat()
        );
        typed_soln = init_guess_typed +
                     (right_precond_ptr->action_inv_M(Q_kry_basis_block*y)).template cast<T>();

    }

    void check_termination() {

        // Check for termination condition with inability to expand subspace if
        // next basis vector is was in the existing Krylov subspace to basis_zero_tol
        int k = kry_space_dim-1;
        if (static_cast<double>(H.get_elem(k+1, k)) <= basis_zero_tol) { this->terminated = true; }

    }

    void typed_iterate() override {
        // Check isn't terminated and that solver isn't attempting to exceed
        // krylov subspace dimension, if is just do nothing
        if (!this->terminated) {
            if (kry_space_dim < max_kry_space_dim) {
                update_subspace_k();
                update_nextq_and_Hkplus1();
                update_QR_fact();
                update_x_minimizing_res();
                check_termination();
            }
        }
    }
    void derived_typed_reset() override { set_initial_space(); } // Set derived reset to erase
                                                                 // current krylov subspace made

public:

    // *** Constructors ***
    GMRESSolve(
        const TypedLinearSystem<M, T> &arg_typed_lin_sys,
        double arg_basis_zero_tol,
        const SolveArgPkg &solve_arg_pkg,
        const PrecondArgPkg<M, W> &precond_arg_pkg = PrecondArgPkg<M, W>()
    ):
        basis_zero_tol(arg_basis_zero_tol),
        left_precond_ptr(precond_arg_pkg.left_precond),
        right_precond_ptr(precond_arg_pkg.right_precond),
        TypedIterativeSolve<M, T>::TypedIterativeSolve(arg_typed_lin_sys, solve_arg_pkg)
    {
        max_iter = (
            (solve_arg_pkg.check_default_max_iter()) ? typed_lin_sys.get_m() : solve_arg_pkg.max_iter
        );
        initializeGMRES();
    }

    // Forbid rvalue instantiation
    GMRESSolve(const TypedLinearSystem<M, T> &&, double, const SolveArgPkg &, const PrecondArgPkg<M, W> &) = delete;
    GMRESSolve(const TypedLinearSystem<M, T> &, double, const SolveArgPkg &&, const PrecondArgPkg<M, W> &) = delete;
    GMRESSolve(const TypedLinearSystem<M, T> &, double, const SolveArgPkg &, const PrecondArgPkg<M, W> &&) = delete;
    GMRESSolve(const TypedLinearSystem<M, T> &&, double, const SolveArgPkg &&, const PrecondArgPkg<M, W> &) = delete;
    GMRESSolve(const TypedLinearSystem<M, T> &, double, const SolveArgPkg &&, const PrecondArgPkg<M, W> &&) = delete;
    GMRESSolve(const TypedLinearSystem<M, T> &&, double, const SolveArgPkg &, const PrecondArgPkg<M, W> &&) = delete;
    GMRESSolve(const TypedLinearSystem<M, T> &&, double, const SolveArgPkg &&, const PrecondArgPkg<M, W> &&) = delete;

};

#endif
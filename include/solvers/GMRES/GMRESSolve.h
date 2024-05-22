#ifndef GMRES_SOLVE_H
#define GMRES_SOLVE_H

#include "../IterativeSolve.h"

#include <chrono>

template <template <typename> typename M, typename T, typename W=T>
class GMRESSolve: public TypedIterativeSolve<M, T>
{
protected:

    std::chrono::high_resolution_clock clock;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;

    void mark_start() {
        start = clock.now();
    }

    void mark_stop(int m) {
        std::cout << "Mark " << m << ": "
                  << std::chrono::duration_cast<std::chrono::microseconds>(clock.now()-start);
    }

    using TypedIterativeSolve<M, T>::typed_lin_sys_ptr;
    using TypedIterativeSolve<M, T>::init_guess_typed;
    using TypedIterativeSolve<M, T>::typed_soln;
    using TypedIterativeSolve<M, T>::max_iter;

    std::shared_ptr<Preconditioner<M, W>> left_precond_ptr;
    std::shared_ptr<Preconditioner<M, W>> right_precond_ptr;

    MatrixDense<T> Q_kry_basis = MatrixDense<T>(cuHandleBundle());
    Vector<T> H_k = Vector<T>(cuHandleBundle());
    MatrixDense<T> H_Q = MatrixDense<T>(cuHandleBundle());
    MatrixDense<T> H_R = MatrixDense<T>(cuHandleBundle());
    Vector<T> next_q = Vector<T>(cuHandleBundle());

    int curr_kry_dim;
    int max_kry_dim;
    double basis_zero_tol;
    Scalar<T> rho;

    void check_compatibility() const {
        if (!left_precond_ptr->check_compatibility_left(typed_lin_sys_ptr->get_m())) {
            throw std::runtime_error("GMRESSolve: Left preconditioner not compatible");
        }
        if (!right_precond_ptr->check_compatibility_right(typed_lin_sys_ptr->get_n())) {
            throw std::runtime_error("GMRESSolve: Right preconditioner not compatible");
        }
    }

    Vector<T> apply_left_precond_A(const Vector<T> &vec) {
        return left_precond_ptr->template casted_action_inv_M<T>(
            (typed_lin_sys_ptr->get_A_typed()*vec).template cast<W>()
        );
    }

    Vector<T> apply_precond_A(const Vector<T> &vec) {
        return apply_left_precond_A(
            right_precond_ptr->template casted_action_inv_M<T>(vec.template cast<W>())
        );
    }

    Vector<T> get_precond_b() {
        return left_precond_ptr->casted_action_inv_M<T>(
            typed_lin_sys_ptr->get_b_typed().template cast<W>()
        );
    }

    Vector<T> calc_precond_residual(const Vector<T> &soln) {
        return(get_precond_b() - apply_left_precond_A(soln));
    }

    void set_initial_space() {

        curr_kry_dim = 0;

        // Pre-allocate all possible space needed to prevent memory
        // re-allocation
        Q_kry_basis = MatrixDense<T>::Zero(
            typed_lin_sys_ptr->get_cu_handles(),
            typed_lin_sys_ptr->get_m(),
            max_kry_dim
        );
        H_k = Vector<T>::Zero(typed_lin_sys_ptr->get_cu_handles(), max_kry_dim+1);
        H_Q = MatrixDense<T>::Identity(
            typed_lin_sys_ptr->get_cu_handles(),
            max_kry_dim+1,
            max_kry_dim+1
        );
        H_R = MatrixDense<T>::Zero(
            typed_lin_sys_ptr->get_cu_handles(),
            max_kry_dim+1,
            max_kry_dim
        );

        // Set rho as initial residual norm
        Vector<T> r_0(calc_precond_residual(init_guess_typed));
        rho = r_0.norm();

        // Initialize next vector q as initial residual and mark as terminated if lucky break
        next_q = r_0;
        if (static_cast<double>(next_q.norm().get_scalar()) <= basis_zero_tol) {
            this->terminated = true;
        }

    }

    void initializeGMRES() {
        if (max_iter > typed_lin_sys_ptr->get_m()) {
            throw std::runtime_error("GMRESSolve: GMRES max_iter exceeds matrix size");
        }
        max_kry_dim = max_iter;
        check_compatibility();
        set_initial_space();
    }

    void update_subspace_k() {

        int curr_kry_idx = curr_kry_dim-1;

        // Normalize next vector q and update subspace with it, assume that
        // checked in previous iteration that vector q was not zero vector
        if (curr_kry_dim == 0) {
            Q_kry_basis.get_col(0).set_from_vec(next_q/next_q.norm());
        } else {
            Q_kry_basis.get_col(curr_kry_idx+1).set_from_vec(next_q/H_k.get_elem(curr_kry_idx+1));
        }
        ++curr_kry_dim;

    }

    void update_nextq_and_Hkplus1() {

        int curr_kry_idx = curr_kry_dim-1;

        // Generate next basis vector base
        next_q = apply_precond_A(Q_kry_basis.get_col(curr_kry_idx));

        // Orthogonalize new basis vector with CGS2
        Vector<T> first_ortho(Q_kry_basis.transpose_prod_subset_cols(0, curr_kry_dim, next_q));
        next_q -= Q_kry_basis.mult_subset_cols(0, curr_kry_dim, first_ortho);
        Vector<T> second_ortho(Q_kry_basis.transpose_prod_subset_cols(0, curr_kry_dim, next_q));
        next_q -= Q_kry_basis.mult_subset_cols(0, curr_kry_dim, second_ortho);

        H_k.set_slice(0, curr_kry_dim, first_ortho + second_ortho);
        H_k.set_elem(curr_kry_idx+1, next_q.norm());

    }

    void update_QR_fact() {

        int curr_kry_idx = curr_kry_dim-1;

        // Initiate next column of QR fact as most recent of H
        H_R.get_col(curr_kry_idx).set_from_vec(H_k);

        // Apply previous Given's rotations to new column
        H_R.get_block(0, curr_kry_idx, curr_kry_idx+1, 1).set_from_vec(
            H_Q.get_block(0, 0, curr_kry_idx+1, curr_kry_idx+1).copy_to_mat().transpose_prod(
                H_k.get_slice(0, curr_kry_idx+1)
            )
        );

        // Apply the final Given's rotation manually making H_R upper triangular
        Scalar<T> alpha = H_R.get_elem(curr_kry_idx, curr_kry_idx);
        Scalar<T> beta = H_R.get_elem(curr_kry_idx+1, curr_kry_idx);
        Scalar<T> r = alpha*alpha + beta*beta;
        r.sqrt();
        Scalar<T> c = alpha/r;
        Scalar<T> minus_s = beta/r;
        H_R.set_elem(curr_kry_idx, curr_kry_idx, r);
        H_R.set_elem(curr_kry_idx+1, curr_kry_idx, SCALAR_ZERO<T>::get());

        Vector<T> H_Q_col_k(H_Q.get_col(curr_kry_idx));
        Vector<T> H_Q_col_kp1(H_Q.get_col(curr_kry_idx+1));
        H_Q.get_col(curr_kry_idx).set_from_vec(H_Q_col_k*c + H_Q_col_kp1*minus_s);
        H_Q.get_col(curr_kry_idx+1).set_from_vec(H_Q_col_kp1*c - H_Q_col_k*minus_s);

    }

    void update_x_minimizing_res() {

        // Calculate rhs to solve
        Vector<T> rho_e1(
            Vector<T>::Zero(typed_lin_sys_ptr->get_cu_handles(), curr_kry_dim+1)
        );
        rho_e1.set_elem(0, rho);
        Vector<T> rhs(
            H_Q.get_block(0, 0, curr_kry_dim+1, curr_kry_dim+1).copy_to_mat().transpose_prod(rho_e1)
        );

        // Use back substitution to solve
        Vector<T> y(
            H_R.get_block(0, 0, curr_kry_dim, curr_kry_dim).copy_to_mat().back_sub(
                rhs.get_slice(0, curr_kry_dim)
            )
        );

        // Update typed_soln adjusting with right preconditioning
        typed_soln = (
            init_guess_typed +
            right_precond_ptr->casted_action_inv_M<T>(
                Q_kry_basis.mult_subset_cols(0, curr_kry_dim, y).template cast<W>()
            )
        );

    }

    void check_termination() {
        int k = curr_kry_dim-1;
        if (static_cast<double>(H_k.get_elem(k+1).get_scalar()) <= basis_zero_tol) {
            this->terminated = true;
        }
    }

    void typed_iterate() override {
        // Check isn't terminated and if exceeding max krylov dim, if is just do nothing
        if (!this->terminated) {
            if (curr_kry_dim < max_kry_dim) {
                // mark_start();
                // update_subspace_k();
                // mark_stop(1); std::cout << " | ";
                // mark_start();
                // update_nextq_and_Hkplus1();
                // mark_stop(2); std::cout << " | ";
                // mark_start();
                // update_QR_fact();
                // mark_stop(3); std::cout << " | ";
                // mark_start();
                // update_x_minimizing_res();
                // mark_stop(4); std::cout << " | ";
                // mark_start();
                // check_termination();
                // mark_stop(5);
                // std::cout << std::endl;
                update_subspace_k();
                update_nextq_and_Hkplus1();
                update_QR_fact();
                update_x_minimizing_res();
                check_termination();
            }
        }
    }

    void derived_typed_reset() override { set_initial_space(); } // Erase current krylov subspace

public:

    // *** Constructors ***
    GMRESSolve(
        const TypedLinearSystem_Intf<M, T> * const arg_typed_lin_sys_ptr,
        double arg_basis_zero_tol,
        const SolveArgPkg &solve_arg_pkg,
        const PrecondArgPkg<M, W> &precond_arg_pkg = PrecondArgPkg<M, W>()
    ):
        basis_zero_tol(arg_basis_zero_tol),
        left_precond_ptr(precond_arg_pkg.left_precond),
        right_precond_ptr(precond_arg_pkg.right_precond),
        TypedIterativeSolve<M, T>::TypedIterativeSolve(arg_typed_lin_sys_ptr, solve_arg_pkg)
    {
        max_iter = (
            (solve_arg_pkg.check_default_max_iter()) ? typed_lin_sys_ptr->get_m() : solve_arg_pkg.max_iter
        );
        initializeGMRES();
    }

    // Forbid rvalue instantiation
    GMRESSolve(const TypedLinearSystem_Intf<M, T> * const, double, const SolveArgPkg &, const PrecondArgPkg<M, W> &&) = delete;
    GMRESSolve(const TypedLinearSystem_Intf<M, T> * const, double, const SolveArgPkg &&, const PrecondArgPkg<M, W> &) = delete;
    GMRESSolve(const TypedLinearSystem_Intf<M, T> * const, double, const SolveArgPkg &&, const PrecondArgPkg<M, W> &&) = delete;

};

#endif
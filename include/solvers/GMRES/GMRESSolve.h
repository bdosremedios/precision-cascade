#ifndef GMRES_SOLVE_H
#define GMRES_SOLVE_H

#include "../IterativeSolve.h"

namespace cascade {

template <template <typename> typename TMatrix, typename TPrecision>
class GMRESSolve: public TypedIterativeSolve<TMatrix, TPrecision>
{
protected:

    using TypedIterativeSolve<TMatrix, TPrecision>::typed_lin_sys_ptr;
    using TypedIterativeSolve<TMatrix, TPrecision>::init_guess_typed;
    using TypedIterativeSolve<TMatrix, TPrecision>::typed_soln;
    using TypedIterativeSolve<TMatrix, TPrecision>::max_iter;

    const PrecondArgPkg<TMatrix, TPrecision> precond_arg_pkg;

    MatrixDense<TPrecision> Q_kry_basis = MatrixDense<TPrecision>(
        cuHandleBundle()
    );
    Vector<TPrecision> H_k = Vector<TPrecision>(
        cuHandleBundle()
    );
    MatrixDense<TPrecision> H_Q = MatrixDense<TPrecision>(
        cuHandleBundle()
    );
    MatrixDense<TPrecision> H_R = MatrixDense<TPrecision>(
        cuHandleBundle()
    );
    Vector<TPrecision> next_q = Vector<TPrecision>(
        cuHandleBundle()
    );

    int curr_kry_dim;
    int max_kry_dim;
    double basis_zero_tol;
    Scalar<TPrecision> rho;

    void check_compatibility() const {
        if (
            !precond_arg_pkg.left_precond->check_compatibility_left(
                typed_lin_sys_ptr->get_m()
            )
        ) {
            throw std::runtime_error(
                "GMRESSolve: Left preconditioner not compatible"
            );
        }
        if (
            !precond_arg_pkg.right_precond->check_compatibility_right(
                typed_lin_sys_ptr->get_n()
            )
        ) {
            throw std::runtime_error(
                "GMRESSolve: Right preconditioner not compatible"
            );
        }
    }

    Vector<TPrecision> apply_left_precond_A(const Vector<TPrecision> &vec) {
        return precond_arg_pkg.left_precond->action_inv_M(
            typed_lin_sys_ptr->get_A_typed()*vec
        );
    }

    Vector<TPrecision> apply_precond_A(const Vector<TPrecision> &vec) {
        return apply_left_precond_A(
            precond_arg_pkg.right_precond->action_inv_M(vec)
        );
    }

    Vector<TPrecision> get_precond_b() {
        return precond_arg_pkg.left_precond->action_inv_M(
            typed_lin_sys_ptr->get_b_typed()
        );
    }

    Vector<TPrecision> calc_precond_residual(const Vector<TPrecision> &soln) {
        return get_precond_b() - apply_left_precond_A(soln);
    }

    void set_initial_space() {

        curr_kry_dim = 0;

        /* Pre-allocate all possible space needed to prevent memory
           re-allocation */
        Q_kry_basis = MatrixDense<TPrecision>::Zero(
            typed_lin_sys_ptr->get_cu_handles(),
            typed_lin_sys_ptr->get_m(),
            max_kry_dim
        );
        H_k = Vector<TPrecision>::Zero(
            typed_lin_sys_ptr->get_cu_handles(),
            max_kry_dim+1
        );
        H_Q = MatrixDense<TPrecision>::Identity(
            typed_lin_sys_ptr->get_cu_handles(),
            max_kry_dim+1,
            max_kry_dim+1
        );
        H_R = MatrixDense<TPrecision>::Zero(
            typed_lin_sys_ptr->get_cu_handles(),
            max_kry_dim+1,
            max_kry_dim
        );

        // Set rho as initial residual norm
        Vector<TPrecision> r_0(calc_precond_residual(init_guess_typed));
        rho = r_0.norm();

        /* Initialize next vector q as initial residual and mark as terminated
           if lucky break */
        next_q = r_0;
        if (static_cast<double>(next_q.norm().get_scalar()) <= basis_zero_tol) {
            this->terminated = true;
        }

    }

    void initializeGMRES() {
        if (max_iter > typed_lin_sys_ptr->get_m()) {
            throw std::runtime_error(
                "GMRESSolve: GMRES max_iter exceeds matrix size"
            );
        }
        max_kry_dim = max_iter;
        check_compatibility();
        set_initial_space();
    }

    void update_subspace_k() {

        int curr_kry_idx = curr_kry_dim-1;

        /* Normalize next vector q and update subspace with it, assume that
           checked in previous iteration that vector q was not zero vector */
        if (curr_kry_dim == 0) {
            Q_kry_basis.get_col(0).set_from_vec(
                next_q/next_q.norm()
            );
        } else {
            Q_kry_basis.get_col(curr_kry_idx+1).set_from_vec(
                next_q/H_k.get_elem(curr_kry_idx+1)
            );
        }
        ++curr_kry_dim;

    }

    void update_nextq_and_Hkplus1() {

        int curr_kry_idx = curr_kry_dim-1;

        // Generate next basis vector base
        next_q = apply_precond_A(
            Q_kry_basis.get_col(curr_kry_idx).copy_to_vec()
        );

        // Orthogonalize new basis vector with CGS2
        Vector<TPrecision> first_ortho(
            Q_kry_basis.transpose_prod_subset_cols(0, curr_kry_dim, next_q)
        );
        next_q -= Q_kry_basis.mult_subset_cols(0, curr_kry_dim, first_ortho);
        Vector<TPrecision> second_ortho(
            Q_kry_basis.transpose_prod_subset_cols(0, curr_kry_dim, next_q)
        );
        next_q -= Q_kry_basis.mult_subset_cols(0, curr_kry_dim, second_ortho);

        H_k.set_slice(0, curr_kry_dim, first_ortho + second_ortho);
        H_k.set_elem(curr_kry_idx+1, next_q.norm());

    }

    void update_QR_fact() {

        int curr_kry_idx = curr_kry_dim-1;

        // Initiate next column of QR fact as most recent of H
        H_R.get_col(curr_kry_idx).set_from_vec(H_k);

        // Apply previous Given's rotations to new column
        H_R.get_block(0, curr_kry_idx, curr_kry_dim, 1).set_from_vec(
            H_Q.get_block(
                0, 0, curr_kry_dim, curr_kry_dim
            ).copy_to_mat().transpose_prod(
                H_k.get_slice(0, curr_kry_dim)
            )
        );

        // Apply the final Given's rotation manually making H_R upper triangular
        Scalar<TPrecision> alpha = H_R.get_elem(curr_kry_idx, curr_kry_idx);
        Scalar<TPrecision> beta = H_R.get_elem(curr_kry_idx+1, curr_kry_idx);
        Scalar<TPrecision> r = alpha*alpha + beta*beta;
        r.sqrt();
        Scalar<TPrecision> c = alpha/r;
        Scalar<TPrecision> minus_s = beta/r;
        H_R.set_elem(curr_kry_idx, curr_kry_idx, r);
        H_R.set_elem(
            curr_kry_idx+1, curr_kry_idx, SCALAR_ZERO<TPrecision>::get()
        );

        Vector<TPrecision> H_Q_col_k(
            H_Q.get_col(curr_kry_idx).copy_to_vec()
        );
        Vector<TPrecision> H_Q_col_kp1(
            H_Q.get_col(curr_kry_idx+1).copy_to_vec()
        );
        H_Q.get_col(curr_kry_idx).set_from_vec(
            H_Q_col_k*c + H_Q_col_kp1*minus_s
        );
        H_Q.get_col(curr_kry_idx+1).set_from_vec(
            H_Q_col_kp1*c - H_Q_col_k*minus_s
        );

    }

    // Virtual for benchmarking override
    virtual void update_x_minimizing_res() {

        // Calculate rhs to solve
        Vector<TPrecision> rho_e1(
            Vector<TPrecision>::Zero(
                typed_lin_sys_ptr->get_cu_handles(), curr_kry_dim+1
            )
        );
        rho_e1.set_elem(0, rho);
        Vector<TPrecision> rhs(
            H_Q.get_block(
                0, 0, curr_kry_dim+1, curr_kry_dim+1
            ).copy_to_mat().transpose_prod(
                rho_e1
            )
        );

        // Use back substitution to solve
        Vector<TPrecision> y(
            H_R.get_block(0, 0, curr_kry_dim, curr_kry_dim).copy_to_mat().back_sub(
                rhs.get_slice(0, curr_kry_dim)
            )
        );

        // Update typed_soln adjusting with right preconditioning
        typed_soln = (
            init_guess_typed +
            precond_arg_pkg.right_precond->action_inv_M(
                Q_kry_basis.mult_subset_cols(0, curr_kry_dim, y)
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
        /* Check isn't terminated and if exceeding max krylov dim, if is just
           do nothing */
        if (!this->terminated) {
            if (curr_kry_dim < max_kry_dim) {
                update_subspace_k();
                update_nextq_and_Hkplus1();
                update_QR_fact();
                update_x_minimizing_res();
                check_termination();
            }
        }
    }

    void derived_typed_reset() override {
        // Erase current krylov subspace
        set_initial_space();
    }

public:

    GMRESSolve(
        const TypedLinearSystem_Intf<TMatrix, TPrecision> * const arg_typed_lin_sys_ptr,
        double arg_basis_zero_tol,
        const SolveArgPkg &arg_solve_arg_pkg,
        const PrecondArgPkg<TMatrix, TPrecision> &arg_precond_arg_pkg = (
            PrecondArgPkg<TMatrix, TPrecision>()
        )
    ):
        basis_zero_tol(arg_basis_zero_tol),
        precond_arg_pkg(arg_precond_arg_pkg),
        TypedIterativeSolve<TMatrix, TPrecision>::TypedIterativeSolve(
            arg_typed_lin_sys_ptr, arg_solve_arg_pkg
        )
    {
        max_iter = (
            (arg_solve_arg_pkg.check_default_max_iter()) ?
            typed_lin_sys_ptr->get_m() : arg_solve_arg_pkg.max_iter
        );
        initializeGMRES();
    }

    // Forbid rvalue instantiation
    GMRESSolve(
        const TypedLinearSystem_Intf<TMatrix, TPrecision> * const, double,
        const SolveArgPkg &&,
        const PrecondArgPkg<TMatrix, TPrecision>
    ) = delete;

};

}

#endif
#ifndef GMRES_SOLVE_H
#define GMRES_SOLVE_H

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
    Vector<T> next_q = Vector<T>(NULL);

    int kry_space_dim;
    int max_kry_space_dim;
    double basis_zero_tol;
    Scalar<T> rho;

    void check_compatibility() const {
        if (!left_precond_ptr->check_compatibility_left(typed_lin_sys.get_m())) {
            throw std::runtime_error("GMRESSolve: Left preconditioner not compatible");
        }
        if (!right_precond_ptr->check_compatibility_right(typed_lin_sys.get_n())) {
            throw std::runtime_error("GMRESSolve: Right preconditioner not compatible");
        }
    }

    Vector<T> apply_left_precond_A(const Vector<T> &vec) {
        return left_precond_ptr->template casted_action_inv_M<T>(
            (typed_lin_sys.get_A_typed()*vec).template cast<W>()
        );
    }

    Vector<T> apply_precond_A(const Vector<T> &vec) {
        return apply_left_precond_A(
            right_precond_ptr->template casted_action_inv_M<T>(vec.template cast<W>())
        );
    }

    Vector<T> get_precond_b() {
        return(
            left_precond_ptr->casted_action_inv_M<T>(
                typed_lin_sys.get_b_typed().template cast<W>()
            )
        );
    }

    Vector<T> calc_precond_residual(const Vector<T> &soln) {
        return(get_precond_b() - apply_left_precond_A(soln));
    }

    void set_initial_space() {

        kry_space_dim = 0;

        // Pre-allocate all possible space needed to prevent memory
        // re-allocation
        Q_kry_basis = MatrixDense<T>::Zero(
            typed_lin_sys.get_handle(),
            typed_lin_sys.get_m(),
            typed_lin_sys.get_m()
        );
        H = MatrixDense<T>::Zero(
            typed_lin_sys.get_handle(),
            typed_lin_sys.get_m()+1,
            typed_lin_sys.get_m()
        );
        Q_H = MatrixDense<T>::Identity(
            typed_lin_sys.get_handle(),
            typed_lin_sys.get_m()+1,
            typed_lin_sys.get_m()+1
        );
        R_H = MatrixDense<T>::Zero(
            typed_lin_sys.get_handle(),
            typed_lin_sys.get_m()+1,
            typed_lin_sys.get_m()
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
        max_kry_space_dim = typed_lin_sys.get_m();
        if (max_iter > max_kry_space_dim) {
            throw std::runtime_error("GMRESSolve: GMRES max_iter exceeds matrix size");
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

        next_q = apply_precond_A(Q_kry_basis.get_col(k));

        // Orthogonlize next_q to previous basis vectors and store coefficients/normalization in H
        Vector<T> H_k(Vector<T>::Zero(typed_lin_sys.get_handle(), typed_lin_sys.get_m()+1));

        for (int i=0; i<=k; ++i) {
            // MGS from newly orthog q used for orthogonalizing next vectors
            Vector<T> q_i(Q_kry_basis.get_col(i));
            H_k.set_elem(i, q_i.dot(next_q));
            next_q -= q_i*H_k.get_elem(i);
        }
        H_k.set_elem(k+1, next_q.norm());
        H.get_col(k).set_from_vec(H_k);

    }

    void update_QR_fact() {

        // Initiate next column of QR fact as most recent of H
        int k = kry_space_dim-1;
        R_H.get_col(k).set_from_vec(H.get_col(k));

        // Apply previous Given's rotations to new column
        R_H.get_block(0, k, k+1, 1).set_from_vec(
            Q_H.get_block(0, 0, k+1, k+1).copy_to_mat().transpose_prod(
                H.get_col(k).copy_to_vec().slice(0, k+1)
            )
        );

        // Apply the final Given's rotation manually making R_H upper triangular
        Scalar<T> alpha = R_H.get_elem(k, k);
        Scalar<T> beta = R_H.get_elem(k+1, k);
        Scalar<T> r = alpha*alpha + beta*beta;
        r.sqrt();
        Scalar<T> c = alpha/r;
        Scalar<T> minus_s = beta/r;
        R_H.set_elem(k, k, r);
        R_H.set_elem(k+1, k, SCALAR_ZERO<T>::get());

        Vector<T> Q_H_col_k(Q_H.get_col(k));
        Vector<T> Q_H_col_kp1(Q_H.get_col(k+1));
        Q_H.get_col(k).set_from_vec(Q_H_col_k*c + Q_H_col_kp1*minus_s);
        Q_H.get_col(k+1).set_from_vec(Q_H_col_kp1*c - Q_H_col_k*minus_s);

    }

    void update_x_minimizing_res() {

        // Calculate rhs to solve
        Vector<T> rho_e1(
            Vector<T>::Zero(typed_lin_sys.get_handle(), kry_space_dim+1)
        );
        rho_e1.set_elem(0, rho);
        Vector<T> rhs(
            Q_H.get_block(0, 0, kry_space_dim+1, kry_space_dim+1).copy_to_mat().transpose_prod(rho_e1)
        );

        // Use back substitution to solve
        Vector<T> y(
            R_H.get_block(0, 0, kry_space_dim, kry_space_dim).copy_to_mat().back_sub(
                rhs.slice(0, kry_space_dim)
            )
        );

        // Update typed_soln adjusting with right preconditioning
        MatrixDense<T> Q_kry_basis_block(
            Q_kry_basis.get_block(0, 0, typed_lin_sys.get_m(), kry_space_dim).copy_to_mat()
        );
        typed_soln = init_guess_typed +
                     right_precond_ptr->casted_action_inv_M<T>(
                        (Q_kry_basis_block*y).template cast<W>()
                     );

    }

    void check_termination() {
        int k = kry_space_dim-1;
        if (static_cast<double>(H.get_elem(k+1, k).get_scalar()) <= basis_zero_tol) {
            this->terminated = true;
        }
    }

    void typed_iterate() override {
        // Check isn't terminated and if exceeding max krylov dim, if is just do nothing
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
    void derived_typed_reset() override { set_initial_space(); } // Erase current krylov subspace

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
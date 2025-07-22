#include "common.hpp"

using namespace dealii;


template <class PreconditionerMp>
class BlockSchurPreconditioner : public Subscriptor
{
  public:
    BlockSchurPreconditioner(double                           gamma,
                             double                           viscosity,
                             const BlockSparseMatrix<double> &S,
                             const SparseMatrix<double>      &P,
                             const PreconditionerMp          &Mppreconditioner);

    void
    vmult(BlockVector<double> &dst, const BlockVector<double> &src) const;

  private:
    const double                     gamma;
    const double                     viscosity;
    const BlockSparseMatrix<double> &stokes_matrix;
    const SparseMatrix<double>      &pressure_mass_matrix;
    const PreconditionerMp          &mp_preconditioner;
    SparseDirectUMFPACK              A_inverse;
};


template <class PreconditionerMp>
BlockSchurPreconditioner<PreconditionerMp>::BlockSchurPreconditioner(
    double                           gamma,
    double                           viscosity,
    const BlockSparseMatrix<double> &S,
    const SparseMatrix<double>      &P,
    const PreconditionerMp          &Mppreconditioner)
    : gamma(gamma)
    , viscosity(viscosity)
    , stokes_matrix(S)
    , pressure_mass_matrix(P)
    , mp_preconditioner(Mppreconditioner)
{
    A_inverse.initialize(stokes_matrix.block(0, 0));
}

template <class PreconditionerMp>
void
BlockSchurPreconditioner<PreconditionerMp>::vmult(
    BlockVector<double>       &dst,
    const BlockVector<double> &src) const
{
    Vector<double> utmp(src.block(0));

    {
        SolverControl solver_control(1000, 1e-6 * src.block(1).l2_norm());
        SolverCG<Vector<double>> cg(solver_control);

        dst.block(1) = 0.0;
        cg.solve(pressure_mass_matrix,
                 dst.block(1),
                 src.block(1),
                 mp_preconditioner);
        dst.block(1) *= -(viscosity + gamma);
    }

    {
        stokes_matrix.block(0, 1).vmult(utmp, dst.block(1));
        utmp *= -1.0;
        utmp += src.block(0);
    }

    A_inverse.vmult(dst.block(0), utmp);
}


template <int dim>
class NavierStokes : public CommonCFD<dim>
{
  public:
    NavierStokes(const Parameters &params,
                 std::string       output_base_name = "ns_solution_");
    void
    run() override;

  protected:
    AffineConstraints<double> zero_constraints;

    BlockSparsityPattern sparsity_pattern;
    SparsityPattern      pressure_sparsity_pattern;

    BlockVector<double> updated_solution;
    BlockVector<double> newton_update;

    BlockSparseMatrix<double> system_matrix;
    SparseMatrix<double>      pressure_mass_matrix;
    double                    gamma = 1.0;

    void
    setup_constraints();
    void
    setup_system_matrix();
    void
    setup_system();

    void
    build_local_matrix(std::vector<double>         &div_phi_u,
                       std::vector<Tensor<2, dim>> &grad_phi_u,
                       std::vector<double>         &phi_p,
                       std::vector<Tensor<1, dim>> &phi_u,
                       Tensor<1, dim>              &velocity_values,
                       Tensor<2, dim>              &velocity_gradients,
                       double                       JxW,
                       const unsigned int           dofs_per_cell,
                       FullMatrix<double>          &local_matrix);

    void
    build_local_rhs(std::vector<double>         &div_phi_u,
                    std::vector<Tensor<2, dim>> &grad_phi_u,
                    std::vector<double>         &phi_p,
                    std::vector<Tensor<1, dim>> &phi_u,
                    Tensor<1, dim>              &velocity_values,
                    Tensor<2, dim>              &velocity_gradients,
                    double                       pressure_value,
                    double                       JxW,
                    const unsigned int           dofs_per_cell,
                    Vector<double>              &local_rhs);

    void
    assemble(const bool initial_step, const bool assemble_matrix);
    unsigned int
    solve(const bool initial_step);

    void
    newton_iteration(const double       tolerance,
                     const unsigned int max_n_line_searches,
                     const bool         is_initial_step);
    void
    assemble_rhs(const bool initial_step);

    void
    assemble_system(const bool initial_step);
};
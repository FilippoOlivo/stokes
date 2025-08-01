#include "common.hpp"

using namespace dealii;

template <class PreconditionerMp>
class BlockSchurPreconditioner : public Subscriptor
{
  public:
    BlockSchurPreconditioner(double gamma,
                             double viscosity,
                             const TrilinosWrappers::BlockSparseMatrix &S,
                             const TrilinosWrappers::SparseMatrix      &P,
                             const PreconditionerMp      &Mppreconditioner,
                             const MPI_Comm              &mpi_communicator,
                             const std::vector<IndexSet> &owned_partitioning);

    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const;

  private:
    const double                               gamma;
    const double                               viscosity;
    const TrilinosWrappers::BlockSparseMatrix &stokes_matrix;
    const TrilinosWrappers::SparseMatrix      &pressure_mass_matrix;
    const PreconditionerMp                    &mp_preconditioner;
    const MPI_Comm                            &mpi_communicator;
    const std::vector<IndexSet>               &owned_partitioning;

    InverseMatrix<TrilinosWrappers::SparseMatrix> A_inverse;
};

template <class PreconditionerMp>
BlockSchurPreconditioner<PreconditionerMp>::BlockSchurPreconditioner(
    double                                     gamma,
    double                                     viscosity,
    const TrilinosWrappers::BlockSparseMatrix &S,
    const TrilinosWrappers::SparseMatrix      &P,
    const PreconditionerMp                    &Mppreconditioner,
    const MPI_Comm                            &mpi_communicator,
    const std::vector<IndexSet>               &owned_partitioning)
    : gamma(gamma)
    , viscosity(viscosity)
    , stokes_matrix(S)
    , pressure_mass_matrix(P)
    , mp_preconditioner(Mppreconditioner)
    , mpi_communicator(mpi_communicator)
    , owned_partitioning(owned_partitioning)
    , A_inverse(stokes_matrix.block(0, 0))
{}

template <class PreconditionerMp>
void
BlockSchurPreconditioner<PreconditionerMp>::vmult(
    TrilinosWrappers::MPI::BlockVector       &dst,
    const TrilinosWrappers::MPI::BlockVector &src) const
{
    // Temporary vector for velocity component
    TrilinosWrappers::MPI::Vector utmp(owned_partitioning[0], mpi_communicator);

    SolverControl solver_control(1000, 1e-6 * src.block(1).l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> cg(solver_control);

    dst.block(1) = 0.0;
    cg.solve(pressure_mass_matrix,
             dst.block(1),
             src.block(1),
             mp_preconditioner);

    dst.block(1) *= -(viscosity + gamma);

    stokes_matrix.block(0, 1).vmult(utmp, dst.block(1));
    utmp *= -1.0;
    utmp += src.block(0);


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

    TrilinosWrappers::MPI::BlockVector old_solution;
    TrilinosWrappers::MPI::BlockVector newton_update;

    TrilinosWrappers::BlockSparseMatrix system_matrix;
    TrilinosWrappers::SparseMatrix      pressure_mass_matrix;
    double                              gamma = 1.0;

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

    double
    compute_residual();

    void
    compute_local_residual(std::vector<double>         &div_phi_u,
                           std::vector<Tensor<2, dim>> &grad_phi_u,
                           std::vector<Tensor<1, dim>> &phi_u,
                           Tensor<1, dim>              &velocity_values,
                           Tensor<2, dim>              &velocity_gradients,
                           double                       pressure_value,
                           double                       JxW,
                           const unsigned int           dofs_per_cell,
                           Vector<double>              &local_rhs);

    void
    assemble(bool initial_guess);

    unsigned int
    solve(TrilinosWrappers::MPI::BlockVector &solution);

    void
    newton_iteration(const double tolerance, const unsigned int max_iterations);

    void
    compute_initial_guess(TrilinosWrappers::MPI::BlockVector &solution);
    void
    build_local_matrix_initial_guess(std::vector<Tensor<2, dim>> &grad_phi_u,
                                     std::vector<double>         &div_phi_u,
                                     std::vector<double>         &phi_p,
                                     double                       JxW,
                                     const unsigned int           dofs_per_cell,
                                     FullMatrix<double>          &local_matrix);
};
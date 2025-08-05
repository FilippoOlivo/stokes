#pragma once
#include "../include/inverse_matrix.hpp"
#include "common.hpp"

using namespace dealii;

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
    unsigned int                        n_it;

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
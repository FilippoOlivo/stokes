#include "../include/BP_stokes.hpp"

template <int dim>
void
StabilizedStokes<dim>::setup_system_matrix()
{
    TimerOutput::Scope timer(this->computing_timer, "setup_system_matrix");
    this->system_matrix.clear();
    // Initialize the sparsity pattern and system matrix
    BlockDynamicSparsityPattern dsp(this->dofs_per_block, this->dofs_per_block);
    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
            coupling[c][d] = DoFTools::always;


    DoFTools::make_sparsity_pattern(
        this->dof_handler, coupling, dsp, this->constraints, false);
    this->sparsity_pattern.copy_from(dsp);
    this->system_matrix.reinit(this->sparsity_pattern);
}

template <int dim>
void
StabilizedStokes<dim>::build_local_matrix(
    std::vector<SymmetricTensor<2, dim>> &symgrad_phi_u,
    std::vector<double>                  &div_phi_u,
    std::vector<Tensor<1, dim>>          &grad_phi_p,
    std::vector<double>                  &phi_p,
    double                                JxW,
    const unsigned int                    dofs_per_cell,
    FullMatrix<double>                   &local_matrix,
    double                                h_k_squared)
{
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j <= i; ++j)
            {
                local_matrix(i, j) +=
                    (2 * this->viscosity *
                         (symgrad_phi_u[i] * symgrad_phi_u[j]) -
                     div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j] +
                     grad_phi_p[i] * grad_phi_p[j] * h_k_squared * delta) *
                    JxW;
                local_matrix(j, i) = local_matrix(i, j); // Ensure symmetry
            }
}

template <int dim>
void
StabilizedStokes<dim>::assemble_system()
{
    TimerOutput::Scope timer(this->computing_timer, "assemble_system");
    this->system_matrix = 0;
    this->system_rhs    = 0;

    // Initialize quadrature formula and FEValues object
    const QGauss<dim> quadrature_formula(this->degree_p + 2);
    FEValues<dim>     fe_values(this->fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                                update_JxW_values | update_gradients);

    // Store the number of degrees of freedom per cell and quadrature points
    const unsigned int dofs_per_cell = this->fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    // Store the local contributions
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_preconditioner_matrix(dofs_per_cell,
                                                   dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    // Store symmetric gradient, divergence, and values of velocity/pressure
    std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
    std::vector<double>                  div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>>          grad_phi_p(dofs_per_cell);
    std::vector<double>                  phi_p(dofs_per_cell);


    for (const auto &cell : this->dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);

            local_matrix                = 0;
            local_preconditioner_matrix = 0;
            local_rhs                   = 0;
            double h_k                  = cell->diameter();
            double h_k_squared          = h_k * h_k;
            for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    // Get symmetric gradient, divergence, and values of
                    // velocity and pressure at quadrature point q
                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                        {
                            symgrad_phi_u[k] =
                                fe_values[velocities].symmetric_gradient(k, q);
                            div_phi_u[k] =
                                fe_values[velocities].divergence(k, q);
                            grad_phi_p[k] = fe_values[pressure].gradient(k, q);
                            phi_p[k]      = fe_values[pressure].value(k, q);
                        }

                    build_local_matrix(symgrad_phi_u,
                                       div_phi_u,
                                       grad_phi_p,
                                       phi_p,
                                       fe_values.JxW(q),
                                       dofs_per_cell,
                                       local_matrix,
                                       h_k_squared);
                    this->build_local_preconditioner_matrix(
                        phi_p,
                        fe_values.JxW(q),
                        dofs_per_cell,
                        local_preconditioner_matrix);
                }
            // Distribute local contributions to global system matrix and rhs
            cell->get_dof_indices(local_dof_indices);
            this->constraints.distribute_local_to_global(local_matrix,
                                                         local_rhs,
                                                         local_dof_indices,
                                                         this->system_matrix,
                                                         this->system_rhs);
            this->constraints.distribute_local_to_global(
                local_preconditioner_matrix,
                local_dof_indices,
                this->preconditioner_matrix);
        }
    // Initialize the preconditioner
    this->A_preconditioner = SparseDirectUMFPACK();
    this->A_preconditioner.initialize(this->system_matrix.block(0, 0),
                                      SparseDirectUMFPACK::AdditionalData());
}

template class StabilizedStokes<2>;
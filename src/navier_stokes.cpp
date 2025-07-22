#include "../include/navier_stokes.hpp"

template <int dim>
NavierStokes<dim>::NavierStokes(const Parameters &params,
                                std::string       output_base_name)

    : CommonCFD<dim>(params, output_base_name){};

template <int dim>
void
NavierStokes<dim>::setup_constraints()
{
    TimerOutput::Scope timer(this->computing_timer, "setup_constraints");
    // Initialize the constraints
    this->constraints.clear();
    this->constraints.reinit();

    const FEValuesExtractors::Vector velocities(0);
    DoFTools::make_hanging_node_constraints(this->dof_handler,
                                            this->constraints);
    VectorTools::interpolate_boundary_values(
        this->dof_handler,
        10,
        InletBoundary(this->params.inlet_velocity),
        this->constraints,
        this->fe.component_mask(velocities));
    VectorTools::interpolate_boundary_values(
        this->dof_handler,
        30,
        Functions::ZeroFunction<dim>(dim + 1),
        this->constraints,
        this->fe.component_mask(velocities));
    VectorTools::interpolate_boundary_values(
        this->dof_handler,
        40,
        Functions::ZeroFunction<dim>(dim + 1),
        this->constraints,
        this->fe.component_mask(velocities));

    const FEValuesExtractors::Scalar pressure(dim);
    VectorTools::interpolate_boundary_values(this->dof_handler,
                                             20,
                                             Functions::ZeroFunction<dim>(dim +
                                                                          1),
                                             this->constraints,
                                             this->fe.component_mask(pressure));
    this->constraints.close();

    zero_constraints.clear();
    zero_constraints.reinit();
    DoFTools::make_hanging_node_constraints(this->dof_handler,
                                            zero_constraints);
    VectorTools::interpolate_boundary_values(
        this->dof_handler,
        10,
        Functions::ZeroFunction<dim>(dim + 1),
        zero_constraints,
        this->fe.component_mask(velocities));
    VectorTools::interpolate_boundary_values(
        this->dof_handler,
        30,
        Functions::ZeroFunction<dim>(dim + 1),
        zero_constraints,
        this->fe.component_mask(velocities));
    VectorTools::interpolate_boundary_values(
        this->dof_handler,
        40,
        Functions::ZeroFunction<dim>(dim + 1),
        zero_constraints,
        this->fe.component_mask(velocities));

    VectorTools::interpolate_boundary_values(this->dof_handler,
                                             20,
                                             Functions::ZeroFunction<dim>(dim +
                                                                          1),
                                             zero_constraints,
                                             this->fe.component_mask(pressure));

    zero_constraints.close();
}

template <int dim>
void
NavierStokes<dim>::setup_system_matrix()
{
    TimerOutput::Scope timer(this->computing_timer, "setup_system_matrix");
    pressure_mass_matrix.clear();
    system_matrix.clear();
    // Initialize the sparsity pattern and system matrix
    BlockDynamicSparsityPattern dsp(this->dofs_per_block, this->dofs_per_block);
    DoFTools::make_sparsity_pattern(this->dof_handler,
                                    dsp,
                                    this->constraints,
                                    false);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
}

template <int dim>
void
NavierStokes<dim>::setup_system()
{
    this->setup_dofhandler();
    setup_constraints();
    setup_system_matrix();

    this->solution.reinit(this->dofs_per_block);
    newton_update.reinit(this->dofs_per_block);
    this->system_rhs.reinit(this->dofs_per_block);
}

template <int dim>
void
NavierStokes<dim>::build_local_matrix(std::vector<double>         &div_phi_u,
                                      std::vector<Tensor<2, dim>> &grad_phi_u,
                                      std::vector<double>         &phi_p,
                                      std::vector<Tensor<1, dim>> &phi_u,
                                      Tensor<1, dim>     &velocity_values,
                                      Tensor<2, dim>     &velocity_gradients,
                                      double              JxW,
                                      const unsigned int  dofs_per_cell,
                                      FullMatrix<double> &local_matrix)
{
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
            local_matrix(i, j) +=
                (this->params.viscosity *
                     scalar_product(grad_phi_u[i], grad_phi_u[j]) +
                 phi_u[i] * (velocity_gradients * phi_u[j]) +
                 phi_u[i] * (grad_phi_u[j] * velocity_values) -
                 div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j] +
                 gamma * div_phi_u[i] * div_phi_u[j] + phi_p[i] * phi_p[j]) *
                JxW;
}

template <int dim>
void
NavierStokes<dim>::build_local_rhs(std::vector<double>         &div_phi_u,
                                   std::vector<Tensor<2, dim>> &grad_phi_u,
                                   std::vector<double>         &phi_p,
                                   std::vector<Tensor<1, dim>> &phi_u,
                                   Tensor<1, dim>              &velocity_values,
                                   Tensor<2, dim>    &velocity_gradients,
                                   double             pressure_value,
                                   double             JxW,
                                   const unsigned int dofs_per_cell,
                                   Vector<double>    &local_rhs)
{
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            double velocity_divergence = trace(velocity_gradients);
            local_rhs(i) +=
                (-this->params.viscosity *
                     scalar_product(grad_phi_u[i], velocity_gradients) -
                 phi_u[i] * (velocity_gradients * velocity_values) +
                 div_phi_u[i] * pressure_value +
                 phi_p[i] * velocity_divergence -
                 gamma * div_phi_u[i] * velocity_divergence) *
                JxW;
        }
}

template <int dim>
void
NavierStokes<dim>::assemble(const bool initial_step, const bool assemble_matrix)
{
    TimerOutput::Scope timer(this->computing_timer, "assemble_system");
    if (assemble_matrix)
        system_matrix = 0;
    this->system_rhs = 0;

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
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    // Store symmetric gradient, divergence, and values of velocity/pressure
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double>         div_phi_u(dofs_per_cell);
    std::vector<double>         phi_p(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);

    std::vector<Tensor<1, dim>> velocity_values(dofs_per_cell);
    std::vector<double>         pressure_values(dofs_per_cell);
    std::vector<Tensor<2, dim>> velocity_gradients(dofs_per_cell);

    for (const auto &cell : this->dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);

            local_matrix = 0;
            local_rhs    = 0;

            // Compute velocity, pressure and velocity gradient values at
            // quadrature points
            fe_values[velocities].get_function_values(updated_solution,
                                                      velocity_values);
            fe_values[pressure].get_function_values(updated_solution,
                                                    pressure_values);
            fe_values[velocities].get_function_gradients(updated_solution,
                                                         velocity_gradients);

            for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    // Get symmetric gradient, divergence, and values of
                    // velocity and pressure at quadrature point q
                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                        {
                            div_phi_u[k] =
                                fe_values[velocities].divergence(k, q);
                            phi_p[k] = fe_values[pressure].value(k, q);
                            grad_phi_u[k] =
                                fe_values[velocities].gradient(k, q);
                            phi_u[k] = fe_values[velocities].value(k, q);
                        }

                    if (assemble_matrix)
                        build_local_matrix(div_phi_u,
                                           grad_phi_u,
                                           phi_p,
                                           phi_u,
                                           velocity_values[q],
                                           velocity_gradients[q],
                                           fe_values.JxW(q),
                                           dofs_per_cell,
                                           local_matrix);
                    build_local_rhs(div_phi_u,
                                    grad_phi_u,
                                    phi_p,
                                    phi_u,
                                    velocity_values[q],
                                    velocity_gradients[q],
                                    pressure_values[q],
                                    fe_values.JxW(q),
                                    dofs_per_cell,
                                    local_rhs);
                }
            cell->get_dof_indices(local_dof_indices);

            const AffineConstraints<double> &constraints_used =
                initial_step ? this->constraints : zero_constraints;

            if (assemble_matrix)
                {
                    constraints_used.distribute_local_to_global(
                        local_matrix,
                        local_rhs,
                        local_dof_indices,
                        system_matrix,
                        this->system_rhs);
                }
            else
                {
                    constraints_used.distribute_local_to_global(
                        local_rhs, local_dof_indices, this->system_rhs);
                }
        }

    if (assemble_matrix)
        {
            pressure_mass_matrix.reinit(sparsity_pattern.block(1, 1));
            pressure_mass_matrix.copy_from(system_matrix.block(1, 1));
            system_matrix.block(1, 1) = 0;
        }
}

template <int dim>
unsigned int
NavierStokes<dim>::solve(const bool initial_step)
{
    const AffineConstraints<double> &constraints_used =
        initial_step ? this->constraints : zero_constraints;

    SolverControl solver_control(system_matrix.m(),
                                 1e-12 * this->system_rhs.l2_norm(),
                                 true);

    SolverFGMRES<BlockVector<double>> gmres(solver_control);
    SparseILU<double>                 pmass_preconditioner;
    pmass_preconditioner.initialize(pressure_mass_matrix,
                                    SparseILU<double>::AdditionalData());

    const BlockSchurPreconditioner<SparseILU<double>> preconditioner(
        gamma,
        this->params.viscosity,
        system_matrix,
        pressure_mass_matrix,
        pmass_preconditioner);
    gmres.solve(system_matrix, newton_update, this->system_rhs, preconditioner);
    constraints_used.distribute(newton_update);
    return solver_control.last_step();
}

template <int dim>
void
NavierStokes<dim>::assemble_rhs(const bool initial_step)
{
    assemble(initial_step, false);
}

template <int dim>
void
NavierStokes<dim>::assemble_system(const bool initial_step)
{
    assemble(initial_step, true);
}

template <int dim>
void
NavierStokes<dim>::newton_iteration(const double       tolerance,
                                    const unsigned int max_n_line_searches,
                                    const bool         is_initial_step)
{
    bool first_step = is_initial_step;

    unsigned int line_search_n = 0;
    double       last_res      = 1.0;
    double       current_res   = 1.0;

    while ((first_step || (current_res > tolerance)) &&
           line_search_n < max_n_line_searches)
        {
            if (first_step)
                {
                    setup_system();
                    updated_solution = this->solution;
                    assemble_system(first_step);
                    solve(first_step);
                    this->solution = newton_update;
                    this->constraints.distribute(this->solution);
                    first_step       = false;
                    updated_solution = this->solution;
                    assemble_rhs(first_step);
                    current_res = this->system_rhs.l2_norm();
                    std::cout << "\tThe residual of initial guess is "
                              << current_res << std::endl;
                    last_res = current_res;
                }
            else
                {
                    updated_solution = this->solution;
                    assemble_system(first_step);
                    unsigned int it = solve(first_step);
                    double       final_alpha;
                    for (double alpha = 1.0; alpha > 1e-5; alpha *= 0.5)
                        {
                            updated_solution = this->solution;
                            updated_solution.add(alpha, newton_update);
                            this->constraints.distribute(updated_solution);
                            assemble_rhs(first_step);
                            current_res = this->system_rhs.l2_norm();
                            if (current_res < last_res)
                                {
                                    final_alpha = alpha;
                                    break;
                                }
                        }
                    std::cout << "\tLine search step " << line_search_n
                              << ": FGMRES iterations = " << it
                              << ", alpha = " << final_alpha
                              << ", residual = " << current_res << std::endl;
                    this->solution = updated_solution;
                    last_res       = current_res;
                    ++line_search_n;
                }
        }
}

template <int dim>
void
NavierStokes<dim>::run()
{
    this->load_grid();
    for (unsigned int cycle = 0; cycle < this->params.refinement_steps; ++cycle)
        {
            if (cycle > 0)
                this->refine_grid();
            std::cout << "Cycle " << cycle << ": "
                      << this->triangulation.n_active_cells() << " active cells"
                      << std::endl;
            newton_iteration(1e-12, 10, true);
            this->output_results(cycle);
        }
}

template class NavierStokes<2>;
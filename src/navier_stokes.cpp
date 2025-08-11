#include "../include/navier_stokes.hpp"

template <int dim>
NavierStokes<dim>::NavierStokes(const Parameters &params,
                                std::string       output_base_name)

    : CommonCFD<dim>(params, output_base_name){};

template <int dim>
void
NavierStokes<dim>::setup_constraints()
{
    // Initialize the constraints
    this->constraints.clear();
    this->constraints.reinit(this->locally_owned_dofs,
                             this->locally_relevant_dofs);

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
}

template <int dim>
void
NavierStokes<dim>::setup_system_matrix()
{
    pressure_mass_matrix.clear();
    system_matrix.clear();
    // Initialize the sparsity pattern and system matrix
    BlockDynamicSparsityPattern dsp(this->relevant_partitioning);
    DoFTools::make_sparsity_pattern(this->dof_handler,
                                    dsp,
                                    this->constraints,
                                    false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               this->locally_owned_dofs,
                                               this->mpi_communicator,
                                               this->locally_relevant_dofs);
    this->sparsity_pattern.copy_from(dsp);
    this->system_matrix.reinit(this->owned_partitioning,
                               this->sparsity_pattern,
                               this->mpi_communicator);
}

template <int dim>
void
NavierStokes<dim>::setup_system()
{
    this->setup_dofhandler();

    setup_constraints();

    setup_system_matrix();

    this->relevant_solution.reinit(this->owned_partitioning,
                                   this->relevant_partitioning,
                                   this->mpi_communicator);
    newton_update.reinit(this->owned_partitioning, this->mpi_communicator);

    this->system_rhs.reinit(this->owned_partitioning, this->mpi_communicator);
}

template <int dim>
void
NavierStokes<dim>::build_local_matrix(std::vector<double> &        div_phi_u,
                                      std::vector<Tensor<2, dim>> &grad_phi_u,
                                      std::vector<double> &        phi_p,
                                      std::vector<Tensor<1, dim>> &phi_u,
                                      Tensor<1, dim> &    velocity_values,
                                      Tensor<2, dim> &    velocity_gradients,
                                      double              JxW,
                                      const unsigned int  dofs_per_cell,
                                      FullMatrix<double> &local_matrix)
{
    double velocity_divergence = trace(velocity_gradients);
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                double a = this->params.viscosity *
                           scalar_product(grad_phi_u[i], grad_phi_u[j]);
                double m = phi_u[i] * (velocity_gradients * phi_u[j]) +
                           0.5 * div_phi_u[j] * velocity_values * phi_u[i];
                double n = phi_u[i] * (grad_phi_u[j] * velocity_values) +
                           0.5 * velocity_divergence * phi_u[i] * phi_u[j];
                double c             = a + m + n;
                double b             = div_phi_u[i] * phi_p[j];
                double b_t           = phi_p[i] * div_phi_u[j];
                double p_p           = phi_p[i] * phi_p[j];
                double stabilization = gamma * div_phi_u[i] * div_phi_u[j];
                local_matrix(i, j) += (c - b - b_t + p_p + stabilization) * JxW;
            }
}

template <int dim>
void
NavierStokes<dim>::compute_local_residual(
    std::vector<double> &        div_phi_u,
    std::vector<Tensor<2, dim>> &grad_phi_u,
    std::vector<Tensor<1, dim>> &phi_u,
    Tensor<1, dim> &             velocity_values,
    Tensor<2, dim> &             velocity_gradients,
    double                       pressure_value,
    double                       JxW,
    const unsigned int           dofs_per_cell,
    Vector<double> &             local_rhs)
{
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            double velocity_divergence = trace(velocity_gradients);

            double viscous_residual =
                this->params.viscosity *
                scalar_product(grad_phi_u[i], velocity_gradients);

            double convective_residual =
                phi_u[i] * (velocity_gradients * velocity_values) +
                0.5 * velocity_divergence * (phi_u[i] * velocity_values);

            double pressure_residual = div_phi_u[i] * pressure_value;

            double stabilization_term =
                gamma * velocity_divergence * div_phi_u[i];

            local_rhs(i) += (-viscous_residual - convective_residual +
                             pressure_residual - stabilization_term) *
                            JxW;
        }
}

template <int dim>
double
NavierStokes<dim>::compute_residual()
{
    TrilinosWrappers::MPI::BlockVector residual;
    residual.reinit(this->owned_partitioning, this->mpi_communicator);
    residual = 0;
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
    Vector<double> local_residual(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    // Store symmetric gradient, divergence, and values of velocity/pressure
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double>         div_phi_u(dofs_per_cell);
    std::vector<double>         phi_p(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);

    std::vector<Tensor<1, dim>> velocity_values(n_q_points);
    std::vector<double>         pressure_values(n_q_points);
    std::vector<Tensor<2, dim>> velocity_gradients(n_q_points);

    for (const auto &cell : this->dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned() == false)
                continue; // Skip artificial cells
            fe_values.reinit(cell);

            local_residual = 0;

            fe_values[velocities].get_function_values(old_solution,
                                                      velocity_values);
            fe_values[pressure].get_function_values(old_solution,
                                                    pressure_values);
            fe_values[velocities].get_function_gradients(old_solution,
                                                         velocity_gradients);
            for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                        {
                            div_phi_u[k] =
                                fe_values[velocities].divergence(k, q);
                            grad_phi_u[k] =
                                fe_values[velocities].gradient(k, q);
                            phi_u[k] = fe_values[velocities].value(k, q);
                            phi_p[k] = fe_values[pressure].value(k, q);
                        }
                    compute_local_residual(div_phi_u,
                                           grad_phi_u,
                                           phi_u,
                                           velocity_values[q],
                                           velocity_gradients[q],
                                           pressure_values[q],
                                           fe_values.JxW(q),
                                           dofs_per_cell,
                                           local_residual);
                }
            cell->get_dof_indices(local_dof_indices);
            this->constraints.distribute_local_to_global(local_residual,
                                                         local_dof_indices,
                                                         residual);
        }
    residual.compress(VectorOperation::add);
    return residual.l2_norm();
}

template <int dim>
void
NavierStokes<dim>::build_local_matrix_initial_guess(
    std::vector<Tensor<2, dim>> &grad_phi_u,
    std::vector<double> &        div_phi_u,
    std::vector<double> &        phi_p,
    double                       JxW,
    const unsigned int           dofs_per_cell,
    FullMatrix<double> &         local_matrix)
{
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j <= i; ++j)
            {
                local_matrix(i, j) +=
                    (this->params.viscosity *
                         scalar_product(grad_phi_u[i], grad_phi_u[j]) -
                     div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j] +
                     phi_p[i] * phi_p[j]) *
                    JxW;
                local_matrix(j, i) = local_matrix(i, j); // Ensure symmetry
            }
}

template <int dim>
void
NavierStokes<dim>::assemble(bool initial_guess)
{
    system_matrix    = 0;
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

    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double>         div_phi_u(dofs_per_cell);
    std::vector<double>         phi_p(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);

    std::vector<Tensor<1, dim>> velocity_values(n_q_points);
    std::vector<Tensor<2, dim>> velocity_gradients(n_q_points);

    for (const auto &cell : this->dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned() == false)
                continue; // Skip artificial cells
            fe_values.reinit(cell);

            local_matrix = 0;
            local_rhs    = 0;

            fe_values[velocities].get_function_values(old_solution,
                                                      velocity_values);
            fe_values[velocities].get_function_gradients(old_solution,
                                                         velocity_gradients);
            for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                        {
                            div_phi_u[k] =
                                fe_values[velocities].divergence(k, q);
                            grad_phi_u[k] =
                                fe_values[velocities].gradient(k, q);
                            if (!initial_guess)
                                phi_u[k] = fe_values[velocities].value(k, q);
                            phi_p[k] = fe_values[pressure].value(k, q);
                        }
                    if (initial_guess)
                        build_local_matrix_initial_guess(grad_phi_u,
                                                         div_phi_u,
                                                         phi_p,
                                                         fe_values.JxW(q),
                                                         dofs_per_cell,
                                                         local_matrix);

                    else
                        build_local_matrix(div_phi_u,
                                           grad_phi_u,
                                           phi_p,
                                           phi_u,
                                           velocity_values[q],
                                           velocity_gradients[q],
                                           fe_values.JxW(q),
                                           dofs_per_cell,
                                           local_matrix);

                    if (!initial_guess)
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            local_rhs(i) += (phi_u[i] * (velocity_gradients[q] *
                                                         velocity_values[q])) *
                                            fe_values.JxW(q);
                }
            cell->get_dof_indices(local_dof_indices);

            this->constraints.distribute_local_to_global(local_matrix,
                                                         local_rhs,
                                                         local_dof_indices,
                                                         system_matrix,
                                                         this->system_rhs);
        }
    system_matrix.compress(VectorOperation::add);
    this->system_rhs.compress(VectorOperation::add);
    pressure_mass_matrix.reinit(sparsity_pattern.block(1, 1));
    pressure_mass_matrix.copy_from(system_matrix.block(1, 1));
    system_matrix.block(1, 1) = 0;
}

template <int dim>
unsigned int
NavierStokes<dim>::solve(TrilinosWrappers::MPI::BlockVector &solution)
{
    this->computing_timer.enter_subsection("initialize solver");
    SolverControl solver_control(system_matrix.m(), 1e-12, true);

    SolverFGMRES<TrilinosWrappers::MPI::BlockVector> gmres(solver_control);
    TrilinosWrappers::PreconditionAMG                pmass_preconditioner;
    pmass_preconditioner.initialize(
        pressure_mass_matrix,
        TrilinosWrappers::PreconditionAMG::AdditionalData());
    this->computing_timer.leave_subsection();

    this->computing_timer.enter_subsection("initialize Schur preconditioner");
    const BlockSchurPreconditioner<TrilinosWrappers::PreconditionAMG>
        preconditioner(this->params.viscosity,
                       system_matrix,
                       pressure_mass_matrix,
                       pmass_preconditioner,
                       this->mpi_communicator,
                       this->owned_partitioning,
                       gamma,
                       true);
    this->computing_timer.leave_subsection();

    this->computing_timer.enter_subsection("solve system");
    gmres.solve(system_matrix, solution, this->system_rhs, preconditioner);
    this->constraints.distribute(solution);
    this->computing_timer.leave_subsection();
    return solver_control.last_step();
}

template <int dim>
void
NavierStokes<dim>::compute_initial_guess(
    TrilinosWrappers::MPI::BlockVector &solution)
{
    this->computing_timer.enter_subsection("assemble initial guess");
    assemble(true);
    this->computing_timer.leave_subsection();
    // this->pcout << "Assembled initial guess." << std::endl;
    this->computing_timer.enter_subsection("init initial guess solver");
    SolverControl solver_control(this->system_matrix.m(), 1e-12, true);
    SolverFGMRES<TrilinosWrappers::MPI::BlockVector> gmres(solver_control);
    TrilinosWrappers::PreconditionAMG                pmass_preconditioner;
    pmass_preconditioner.initialize(
        pressure_mass_matrix,
        TrilinosWrappers::PreconditionAMG::AdditionalData());

    const BlockSchurPreconditioner<TrilinosWrappers::PreconditionAMG>
        preconditioner(this->params.viscosity,
                       this->system_matrix,
                       pressure_mass_matrix,
                       pmass_preconditioner,
                       this->mpi_communicator,
                       this->owned_partitioning,
                       0.0,
                       false);
    this->computing_timer.leave_subsection();
    // this->pcout << "Initialized initial guess solver." << std::endl;
    this->computing_timer.enter_subsection("solve initial guess");
    gmres.solve(this->system_matrix,
                solution,
                this->system_rhs,
                preconditioner);
    this->constraints.distribute(solution);
    this->computing_timer.leave_subsection();
    this->pcout << "Initial guess computed: " << std::endl
                << "\tFGMRES iterations = " << solver_control.last_step()
                << std::endl;
}

template <int dim>
void
NavierStokes<dim>::newton_iteration(const double       tolerance,
                                    const unsigned int max_iterations)
{
    unsigned int                       line_search_n    = 0;
    double                             current_residual = 0.0;
    double                             past_residual    = 1.0;
    TrilinosWrappers::MPI::BlockVector solution;
    solution.reinit(this->owned_partitioning, this->mpi_communicator);
    old_solution.reinit(this->owned_partitioning,
                        this->relevant_partitioning,
                        this->mpi_communicator);
    TrilinosWrappers::MPI::BlockVector initial_guess;
    initial_guess.reinit(this->owned_partitioning, this->mpi_communicator);
    // this->pcout << "Starting Newton iteration." << std::endl;
    compute_initial_guess(initial_guess);
    old_solution = initial_guess;
    n_it         = 0;
    while (past_residual > tolerance && line_search_n < max_iterations)
        {
            this->computing_timer.enter_subsection("assemble");
            assemble(false);
            this->computing_timer.leave_subsection();

            unsigned int it;
            it           = solve(solution);
            old_solution = solution;

            this->computing_timer.enter_subsection("compute_residual");
            current_residual = compute_residual();
            this->computing_timer.leave_subsection();

            this->pcout << "\tLine search step " << line_search_n
                        << ": FGMRES iterations = " << it
                        << ", residual = " << current_residual << std::endl;
            ++line_search_n;
            if (current_residual > past_residual)
                {
                    this->pcout << "Warning: residual increased from "
                                << past_residual << " to " << current_residual
                                << ". Stopping iteration." << std::endl;
                    break;
                }
            past_residual = current_residual;
            ++n_it;
        }
    ++n_it;
    this->relevant_solution = old_solution;
}

template <int dim>
void
NavierStokes<dim>::run()
{
    this->load_grid();
    this->pcout << "Loaded grid." << std::endl;
    for (unsigned int cycle = 0; cycle < this->params.refinement_steps; ++cycle)
        {
            if (cycle > 0)
                this->refine_grid();
            this->computing_timer.reset();
            Timer total_timer;
            total_timer.start();

            this->computing_timer.enter_subsection("setup_system");
            setup_system();
            this->computing_timer.leave_subsection();
            newton_iteration(1e-12, 5);

            this->computing_timer.enter_subsection("output_results");
            this->output_results(cycle);
            this->computing_timer.leave_subsection();

            total_timer.stop();
            const double wall_time = total_timer.wall_time();
            this->computing_timer.print_summary();
            this->write_timer_to_csv(wall_time, n_it);
            this->pcout << std::endl;
        }
}

template class NavierStokes<2>;
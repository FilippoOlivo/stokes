#pragma once
#include "common.hpp"

using namespace dealii;

template <int dim>
class BaseStokes : public CommonCFD<dim>
{
  public:
    BaseStokes(const Parameters &params, std::string output_base_name);
    void
    run() override;
    TrilinosWrappers::MPI::BlockVector
    get_solution();

  protected:
    BlockSparsityPattern           sparsity_pattern;
    BlockSparsityPattern           preconditioner_sparsity_pattern;
    TrilinosWrappers::SparseMatrix pressure_mass_matrix;

    void
    setup_constraints();
    virtual void
    setup_system_matrix() = 0;
    void
    setup_system();
    virtual void
    assemble_system() = 0;
    void
    solve();
};

template <int dim>
BaseStokes<dim>::BaseStokes(const Parameters &params,
                            std::string       output_base_name)

    : CommonCFD<dim>(params, output_base_name){};

template <int dim>
void
BaseStokes<dim>::setup_constraints()
{
    // Initialize the this->constraints
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

    const FEValuesExtractors::Vector pressure(dim - 1);
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
BaseStokes<dim>::setup_system()
{
    this->setup_dofhandler();
    setup_constraints();
    setup_system_matrix();
    // Initialize the system right-hand side vector
    this->relevant_solution.reinit(this->owned_partitioning,
                                   this->relevant_partitioning,
                                   this->mpi_communicator);
    // Initialize the system right-hand side vector
    this->system_rhs.reinit(this->owned_partitioning, this->mpi_communicator);
}

template <int dim>
void
BaseStokes<dim>::solve()
{
    TrilinosWrappers::MPI::BlockVector solution(this->owned_partitioning,
                                                this->mpi_communicator);
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
                       this->owned_partitioning);

    gmres.solve(this->system_matrix,
                solution,
                this->system_rhs,
                preconditioner);
    this->constraints.distribute(solution);
    this->relevant_solution = solution;
    this->pcout << "  " << solver_control.last_step()
                << " outer CG Schur complement iterations for pressure"
                << std::endl;
}


template <int dim>
void
BaseStokes<dim>::run()
{
    this->load_grid();

    for (unsigned int cycle = 0; cycle < this->params.refinement_steps; ++cycle)
        {
            if (cycle > 0)
                this->refine_grid();
            setup_system();
            assemble_system();
            solve();
            this->computing_timer.print_summary();
            this->computing_timer.reset();
            this->pcout << std::endl;
            this->output_results(0);
        }
}
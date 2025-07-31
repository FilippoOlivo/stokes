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
    BlockSparsityPattern                sparsity_pattern;
    BlockSparsityPattern                preconditioner_sparsity_pattern;
    TrilinosWrappers::BlockSparseMatrix preconditioner_matrix;

    TrilinosWrappers::PreconditionChebyshev A_preconditioner;

    void
    setup_constraints();
    virtual void
    setup_system_matrix() = 0;
    void
    setup_preconditioner_matrix();
    void
    setup_system();

    void
    build_local_preconditioner_matrix(
        std::vector<double> &phi_p,
        double               JxW,
        const unsigned int   dofs_per_cell,
        FullMatrix<double>  &local_preconditioner_matrix);
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
BaseStokes<dim>::setup_preconditioner_matrix()
{
    preconditioner_matrix.clear();
    A_preconditioner.clear();

    // Initialize the sparsity pattern and preconditioner matrix
    Table<2, DoFTools::Coupling> preconditioner_coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
            if (((c == dim) && (d == dim)))
                preconditioner_coupling[c][d] = DoFTools::always;
            else
                preconditioner_coupling[c][d] = DoFTools::none;
    BlockDynamicSparsityPattern preconditioner_dsp(this->relevant_partitioning);
    DoFTools::make_sparsity_pattern(this->dof_handler,
                                    preconditioner_coupling,
                                    preconditioner_dsp,
                                    this->constraints,
                                    false);
    SparsityTools::distribute_sparsity_pattern(preconditioner_dsp,
                                               this->locally_owned_dofs,
                                               this->mpi_communicator,
                                               this->locally_relevant_dofs);

    preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);
    preconditioner_matrix.reinit(this->owned_partitioning,
                                 preconditioner_sparsity_pattern,
                                 this->mpi_communicator);
}

template <int dim>
void
BaseStokes<dim>::setup_system()
{
    TimerOutput::Scope t_dof(this->computing_timer, "setup_dofhandler");
    {
        this->setup_dofhandler();
    }
    TimerOutput::Scope t_constraints(this->computing_timer,
                                     "setup_constraints");
    {
        setup_constraints();
    }
    TimerOutput::Scope t_system(this->computing_timer, "setup_system_matrix");
    {
        setup_system_matrix();
    }
    TimerOutput::Scope t_prec(this->computing_timer,
                              "setup_preconditioner_matrix");
    {
        setup_preconditioner_matrix();
    }

    TimerOutput::Scope t_solution(this->computing_timer, "setup_solution");
    {
        // Initialize the system right-hand side vector
        this->relevant_solution.reinit(this->owned_partitioning,
                                       this->relevant_partitioning,
                                       this->mpi_communicator);
    }
    TimerOutput::Scope t_rhs(this->computing_timer, "setup_system_rhs");
    {
        // Initialize the system right-hand side vector
        this->system_rhs.reinit(this->owned_partitioning,
                                this->mpi_communicator);
    }
}

template <int dim>
void
BaseStokes<dim>::build_local_preconditioner_matrix(
    std::vector<double> &phi_p,
    double               JxW,
    const unsigned int   dofs_per_cell,
    FullMatrix<double>  &local_preconditioner_matrix)
{
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j <= i; ++j)
            {
                local_preconditioner_matrix(i, j) +=
                    (phi_p[i] * phi_p[j]) * JxW;
                local_preconditioner_matrix(j, i) =
                    local_preconditioner_matrix(i, j); // Ensure symmetry
            }
}

template <int dim>
void
BaseStokes<dim>::solve()
{
    TrilinosWrappers::MPI::BlockVector solution(this->owned_partitioning,
                                                this->mpi_communicator);
    const InverseMatrix<TrilinosWrappers::SparseMatrix,
                        TrilinosWrappers::PreconditionChebyshev>
        A_inverse(this->system_matrix.block(0, 0), A_preconditioner);
    TrilinosWrappers::MPI::Vector tmp(this->owned_partitioning[0],
                                      this->mpi_communicator);

    TimerOutput::Scope t(this->computing_timer, "solve_pressure");
    {
        TrilinosWrappers::MPI::Vector schur_rhs(this->owned_partitioning[1],
                                                this->mpi_communicator);
        A_inverse.vmult(tmp, this->system_rhs.block(0));

        this->system_matrix.block(1, 0).vmult(schur_rhs, tmp);
        schur_rhs -= this->system_rhs.block(1);

        SchurComplement<TrilinosWrappers::PreconditionChebyshev>
            schur_complement(this->system_matrix,
                             A_inverse,
                             this->owned_partitioning,
                             this->mpi_communicator);

        SolverControl solver_control(solution.block(1).size(),
                                     1e-12 * schur_rhs.l2_norm());
        SolverCG<TrilinosWrappers::MPI::Vector> cg(solver_control);
        TrilinosWrappers::PreconditionChebyshev preconditioner;
        preconditioner.initialize(
            this->preconditioner_matrix.block(1, 1),
            TrilinosWrappers::PreconditionChebyshev::AdditionalData());

        InverseMatrix<TrilinosWrappers::SparseMatrix,
                      TrilinosWrappers::PreconditionChebyshev>
            m_inverse(this->preconditioner_matrix.block(1, 1), preconditioner);

        cg.solve(schur_complement, solution.block(1), schur_rhs, m_inverse);

        this->constraints.distribute(solution);

        this->pcout << "  " << solver_control.last_step()
                    << " outer CG Schur complement iterations for pressure"
                    << std::endl;
    }

    TimerOutput::Scope t_velocity(this->computing_timer, "solve_velocity");
    {
        this->system_matrix.block(0, 1).vmult(tmp, solution.block(1));
        tmp *= -1;
        tmp += this->system_rhs.block(0);

        A_inverse.vmult(solution.block(0), tmp);
    }
    TimerOutput::Scope t_constraints(this->computing_timer,
                                     "distribute constraints");
    {
        this->constraints.distribute(solution);
    }
    TimerOutput::Scope t_relevant(this->computing_timer,
                                  "update relevant solution");
    {
        // Update the relevant solution
        this->relevant_solution = solution;
    }
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
            // this->write_timer_to_csv();
            this->computing_timer.print_summary();
            this->computing_timer.reset();
            this->pcout << std::endl;
            this->output_results(0);
        }
}

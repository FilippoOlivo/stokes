#include "common.hpp"

using namespace dealii;

template <int dim>
class BaseStokes : public CommonCFD<dim>
{
  public:
    BaseStokes(const Parameters &params, std::string output_base_name);
    void
    run() override;
    BlockVector<double>
    get_solution();

  protected:
    BlockSparsityPattern sparsity_pattern;
    BlockSparsityPattern preconditioner_sparsity_pattern;

    BlockSparseMatrix<double> system_matrix;
    BlockSparseMatrix<double> preconditioner_matrix;

    SparseDirectUMFPACK A_preconditioner;

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
    TimerOutput::Scope timer(this->computing_timer, "setup_constraints");
    // Initialize the this->constraints
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
    TimerOutput::Scope timer(this->computing_timer,
                             "setup_preconditioner_matrix");
    preconditioner_matrix.clear();

    // Initialize sparsity pattern and preconditioner matrix
    BlockDynamicSparsityPattern preconditioner_dsp(this->dofs_per_block,
                                                   this->dofs_per_block);

    Table<2, DoFTools::Coupling> preconditioner_coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
            if (((c == dim) && (d == dim)))
                preconditioner_coupling[c][d] = DoFTools::always;
            else
                preconditioner_coupling[c][d] = DoFTools::none;

    DoFTools::make_sparsity_pattern(this->dof_handler,
                                    preconditioner_coupling,
                                    preconditioner_dsp,
                                    this->constraints,
                                    false);
    preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);
    preconditioner_matrix.reinit(preconditioner_sparsity_pattern);
}

template <int dim>
void
BaseStokes<dim>::setup_system()
{
    this->setup_dofhandler();
    setup_constraints();
    setup_system_matrix();
    setup_preconditioner_matrix();

    TimerOutput::Scope timer(this->computing_timer, "setup_solution_vectors");
    {
        this->solution.reinit(this->dofs_per_block);
    }
    TimerOutput::Scope timer_rhs(this->computing_timer, "setup_system_rhs");
    {
        this->system_rhs.reinit(this->dofs_per_block);
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
    const InverseMatrix<SparseMatrix<double>, SparseDirectUMFPACK> A_inverse(
        system_matrix.block(0, 0), A_preconditioner);
    Vector<double> tmp(this->solution.block(0).size());

    TimerOutput::Scope timer(this->computing_timer, "solve_pressure");
    {
        Vector<double> schur_rhs(this->solution.block(1).size());
        A_inverse.vmult(tmp, this->system_rhs.block(0));
        system_matrix.block(1, 0).vmult(schur_rhs, tmp);
        schur_rhs -= this->system_rhs.block(1);

        SchurComplement<SparseDirectUMFPACK> schur_complement(system_matrix,
                                                              A_inverse);
        std::cout << "  Solving Schur complement for pressure" << std::endl;
        SolverControl solver_control(10000, 1e-12 * schur_rhs.l2_norm());
        SolverCG<Vector<double>> cg(solver_control);

        SparseILU<double> preconditioner;
        preconditioner.initialize(preconditioner_matrix.block(1, 1),
                                  SparseILU<double>::AdditionalData());

        InverseMatrix<SparseMatrix<double>, SparseILU<double>> m_inverse(
            preconditioner_matrix.block(1, 1), preconditioner);

        cg.solve(schur_complement,
                 this->solution.block(1),
                 schur_rhs,
                 m_inverse);

        this->constraints.distribute(this->solution);

        std::cout << "  " << solver_control.last_step()
                  << " outer CG Schur complement iterations for pressure"
                  << std::endl;
    }
    TimerOutput::Scope t2(this->computing_timer, "solve_velocity");
    {
        system_matrix.block(0, 1).vmult(tmp, this->solution.block(1));
        tmp *= -1;
        tmp += this->system_rhs.block(0);

        A_inverse.vmult(this->solution.block(0), tmp);

        this->constraints.distribute(this->solution);
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
            this->computing_timer.print_summary();
            this->computing_timer.reset();
            std::cout << std::endl;
            this->output_results(cycle);
        }
}

#include <deal.II/base/config.h>

#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include "../include/boundary.hpp"
#include "../include/schur_complement.hpp"

using namespace dealii;

template <int dim>
class BaseStokes
{
  public:
    BaseStokes(unsigned int degree_p,
               unsigned int degree_u,
               std::string  output_base_name);
    void
    run();

  protected:
    std::string         mesh_file = "../mesh.msh";
    unsigned int        degree_p;
    unsigned int        degree_u;
    Triangulation<dim>  triangulation;
    DoFHandler<dim>     dof_handler;
    const FESystem<dim> fe;
    TimerOutput         computing_timer;
    std::string         output_base_name;


    std::vector<types::global_dof_index> dofs_per_block;
    AffineConstraints<double>            constraints;

    BlockSparsityPattern sparsity_pattern;
    BlockSparsityPattern preconditioner_sparsity_pattern;

    BlockVector<double> solution;
    BlockVector<double> system_rhs;

    BlockSparseMatrix<double> system_matrix;
    BlockSparseMatrix<double> preconditioner_matrix;

    SparseDirectUMFPACK A_preconditioner;

    double viscosity = 0.1; // Oil viscosity

    void
    load_grid();

    void
    setup_dofhandler();
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
    void
    output_results(unsigned int cycle);
    void
    refine_grid();
};

template <int dim>
BaseStokes<dim>::BaseStokes(unsigned int degree_p,
                            unsigned int degree_u,
                            std::string  output_base_name)

    : degree_p(degree_p)
    , degree_u(degree_u)
    , dof_handler(triangulation)
    , fe(FE_Q<dim>(degree_u) ^ dim, FE_Q<dim>(degree_p))
    , computing_timer(std::cout, TimerOutput::never, TimerOutput::wall_times)
    , output_base_name(output_base_name){};

template <int dim>
void
BaseStokes<dim>::load_grid()
{
    GridIn<2> grid_in;
    grid_in.attach_triangulation(triangulation);
    std::ifstream input_file(mesh_file);
    grid_in.read_msh(input_file);
}

template <int dim>
void
BaseStokes<dim>::setup_dofhandler()
{
    TimerOutput::Scope timer(computing_timer, "setup_dofhandler");
    dof_handler.distribute_dofs(fe);
    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;
    DoFRenumbering::Cuthill_McKee(dof_handler);

    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

    dofs_per_block =
        DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
}

template <int dim>
void
BaseStokes<dim>::setup_constraints()
{
    TimerOutput::Scope timer(computing_timer, "setup_constraints");
    // Initialize the constraints
    constraints.clear();
    constraints.reinit();

    const FEValuesExtractors::Vector velocities(0);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             10,
                                             InletBoundary(),
                                             constraints,
                                             fe.component_mask(velocities));
    VectorTools::interpolate_boundary_values(dof_handler,
                                             30,
                                             Functions::ZeroFunction<dim>(dim +
                                                                          1),
                                             constraints,
                                             fe.component_mask(velocities));
    VectorTools::interpolate_boundary_values(dof_handler,
                                             40,
                                             Functions::ZeroFunction<dim>(dim +
                                                                          1),
                                             constraints,
                                             fe.component_mask(velocities));

    const FEValuesExtractors::Vector pressure(dim - 1);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             20,
                                             Functions::ZeroFunction<dim>(dim +
                                                                          1),
                                             constraints,
                                             fe.component_mask(pressure));

    constraints.close();
}

template <int dim>
void
BaseStokes<dim>::setup_preconditioner_matrix()
{
    TimerOutput::Scope timer(computing_timer, "setup_preconditioner_matrix");
    preconditioner_matrix.clear();

    // Initialize sparsity pattern and preconditioner matrix
    BlockDynamicSparsityPattern preconditioner_dsp(dofs_per_block,
                                                   dofs_per_block);

    Table<2, DoFTools::Coupling> preconditioner_coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
            if (((c == dim) && (d == dim)))
                preconditioner_coupling[c][d] = DoFTools::always;
            else
                preconditioner_coupling[c][d] = DoFTools::none;

    DoFTools::make_sparsity_pattern(dof_handler,
                                    preconditioner_coupling,
                                    preconditioner_dsp,
                                    constraints,
                                    false);
    preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);
    preconditioner_matrix.reinit(preconditioner_sparsity_pattern);
}

template <int dim>
void
BaseStokes<dim>::setup_system()
{
    setup_dofhandler();
    setup_constraints();
    setup_system_matrix();
    setup_preconditioner_matrix();

    TimerOutput::Scope timer(computing_timer, "setup_solution_vectors");
    {
        solution.reinit(dofs_per_block);
    }
    TimerOutput::Scope timer_rhs(computing_timer, "setup_system_rhs");
    {
        system_rhs.reinit(dofs_per_block);
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
    Vector<double> tmp(solution.block(0).size());

    TimerOutput::Scope timer(computing_timer, "solve_pressure");
    {
        Vector<double> schur_rhs(solution.block(1).size());
        A_inverse.vmult(tmp, system_rhs.block(0));
        system_matrix.block(1, 0).vmult(schur_rhs, tmp);
        schur_rhs -= system_rhs.block(1);

        SchurComplement<SparseDirectUMFPACK> schur_complement(system_matrix,
                                                              A_inverse);

        SolverControl            solver_control(solution.block(1).size(),
                                     1e-12 * schur_rhs.l2_norm());
        SolverCG<Vector<double>> cg(solver_control);

        SparseILU<double> preconditioner;
        preconditioner.initialize(preconditioner_matrix.block(1, 1),
                                  SparseILU<double>::AdditionalData());

        InverseMatrix<SparseMatrix<double>, SparseILU<double>> m_inverse(
            preconditioner_matrix.block(1, 1), preconditioner);

        cg.solve(schur_complement, solution.block(1), schur_rhs, m_inverse);

        constraints.distribute(solution);

        std::cout << "  " << solver_control.last_step()
                  << " outer CG Schur complement iterations for pressure"
                  << std::endl;
    }
    TimerOutput::Scope t2(computing_timer, "solve_velocity");
    {
        system_matrix.block(0, 1).vmult(tmp, solution.block(1));
        tmp *= -1;
        tmp += system_rhs.block(0);

        A_inverse.vmult(solution.block(0), tmp);

        constraints.distribute(solution);
    }
}

template <int dim>
void
BaseStokes<dim>::output_results(unsigned int cycle)
{
    std::vector<std::string> solution_names(2, "velocity");
    solution_names.emplace_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
            2, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);

    DataOut<2> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             solution_names,
                             DataOut<2>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();

    std::ofstream output(output_base_name + std::to_string(cycle) + ".vtk");
    data_out.write_vtk(output);
}

template <int dim>
void
BaseStokes<dim>::refine_grid()
{
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    const FEValuesExtractors::Scalar velocities(0);
    KellyErrorEstimator<dim>::estimate(
        dof_handler,
        QGauss<dim - 1>(degree_p + 1),
        std::map<types::boundary_id, const Function<dim> *>(),
        solution,
        estimated_error_per_cell,
        fe.component_mask(velocities));

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    0.2,
                                                    0.1);
    triangulation.execute_coarsening_and_refinement();
}

template <int dim>
void
BaseStokes<dim>::run()
{
    load_grid();

    for (unsigned int cycle = 0; cycle < 5; ++cycle)
        {
            if (cycle > 0)
                refine_grid();
            setup_system();
            assemble_system();
            solve();
            computing_timer.print_summary();
            computing_timer.reset();
            std::cout << std::endl;
            output_results(cycle);
        }
}
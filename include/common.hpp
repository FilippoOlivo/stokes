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
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include "../include/boundary.hpp"
#include "../include/parameters.hpp"
#include "../include/schur_complement.hpp"

using namespace dealii;

template <int dim>
class CommonCFD
{
  public:
    CommonCFD(const Parameters &params, std::string output_base_name);
    virtual void
    run() = 0;
    BlockVector<double>
    get_solution();


  protected:
    const Parameters   &params;
    unsigned int        degree_p;
    unsigned int        degree_u;
    Triangulation<dim>  triangulation;
    DoFHandler<dim>     dof_handler;
    const FESystem<dim> fe;
    TimerOutput         computing_timer;
    std::string         mesh_file;
    std::string         output_base_name;

    std::vector<types::global_dof_index> dofs_per_block;
    AffineConstraints<double>            constraints;
    BlockSparsityPattern                 sparsity_pattern;
    BlockVector<double>                  solution;
    BlockVector<double>                  system_rhs;
    BlockSparseMatrix<double>            system_matrix;


    void
    setup_dofhandler();
    void
    load_grid();
    void
    output_results(unsigned int cycle);
    void
    refine_grid();
};

template <int dim>
CommonCFD<dim>::CommonCFD(const Parameters &params,
                          std::string       output_base_name)
    : params(params)
    , degree_p(params.degree_p)
    , degree_u(params.degree_u)
    , dof_handler(triangulation)
    , fe(FE_Q<dim>(degree_u) ^ dim, FE_Q<dim>(degree_p))
    , computing_timer(std::cout, TimerOutput::never, TimerOutput::wall_times)
    , mesh_file(params.mesh_file)
    , output_base_name(output_base_name){};

template <int dim>
void
CommonCFD<dim>::setup_dofhandler()
{
    TimerOutput::Scope timer(this->computing_timer, "setup_dofhandler");
    this->dof_handler.distribute_dofs(this->fe);
    std::cout << "Number of degrees of freedom: " << this->dof_handler.n_dofs()
              << std::endl;
    DoFRenumbering::Cuthill_McKee(this->dof_handler);

    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(this->dof_handler, block_component);

    dofs_per_block =
        DoFTools::count_dofs_per_fe_block(this->dof_handler, block_component);
}

template <int dim>
void
CommonCFD<dim>::load_grid()
{
    GridIn<2> grid_in;
    grid_in.attach_triangulation(triangulation);
    std::ifstream input_file(mesh_file);
    grid_in.read_msh(input_file);
}

template <int dim>
void
CommonCFD<dim>::output_results(unsigned int cycle)
{
    std::vector<std::string> solution_names(2, "velocity");
    solution_names.emplace_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
            2, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);

    DataOut<2> data_out;
    data_out.attach_dof_handler(this->dof_handler);
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
CommonCFD<dim>::refine_grid()
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
BlockVector<double>
CommonCFD<dim>::get_solution()
{
    return solution;
}
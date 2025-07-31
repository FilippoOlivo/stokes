#include <deal.II/base/config.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

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
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
// #include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
// #include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>

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
    const Parameters                         &params;
    MPI_Comm                                  mpi_communicator;
    unsigned int                              mpi_size;
    unsigned int                              mpi_rank;
    unsigned int                              degree_p;
    unsigned int                              degree_u;
    parallel::distributed::Triangulation<dim> triangulation;
    DoFHandler<dim>                           dof_handler;
    const FESystem<dim>                       fe;
    ConditionalOStream                        pcout;
    TimerOutput                               computing_timer;
    std::string                               mesh_file;
    std::string                               output_base_name;


    std::vector<types::global_dof_index> dofs_per_block;
    AffineConstraints<double>            constraints;
    BlockSparsityPattern                 sparsity_pattern;
    TrilinosWrappers::MPI::BlockVector   relevant_solution;
    TrilinosWrappers::MPI::BlockVector   system_rhs;
    TrilinosWrappers::BlockSparseMatrix  system_matrix;

    IndexSet              locally_owned_dofs;
    IndexSet              locally_relevant_dofs;
    std::vector<IndexSet> owned_partitioning;
    std::vector<IndexSet> relevant_partitioning;


    void
    setup_dofhandler();
    void
    load_grid();
    void
    output_results(unsigned int cycle);
    void
    refine_grid();
    void
    write_timer_to_csv(const double total_time);
};

template <int dim>
CommonCFD<dim>::CommonCFD(const Parameters &params,
                          std::string       output_base_name)
    : params(params)
    , mpi_communicator(MPI_COMM_WORLD)
    , mpi_size(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , mpi_rank(Utilities::MPI::this_mpi_process(mpi_communicator))
    , degree_p(params.degree_p)
    , degree_u(params.degree_u)
    , triangulation(mpi_communicator)
    , dof_handler(triangulation)
    , fe(FE_Q<dim>(degree_u) ^ dim, FE_Q<dim>(degree_p))
    , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    , computing_timer(mpi_communicator,       // MPI communicator
                      pcout,                  // ConditionalOStream
                      TimerOutput::never,     // no output during run
                      TimerOutput::wall_times // measure wall times
                      )
    , mesh_file(params.mesh_file)
    , output_base_name(output_base_name){};

template <int dim>
void
CommonCFD<dim>::setup_dofhandler()
{
    dof_handler.distribute_dofs(fe);
    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

    const std::vector<types::global_dof_index> dofs_per_block =
        DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    owned_partitioning = {locally_owned_dofs.get_view(0, n_u),
                          locally_owned_dofs.get_view(n_u, n_u + n_p)};

    locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);
    relevant_partitioning = {locally_relevant_dofs.get_view(0, n_u),
                             locally_relevant_dofs.get_view(n_u, n_u + n_p)};
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
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
            dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(relevant_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    std::string filename = (output_base_name + std::to_string(cycle) + ".vtu");
    data_out.write_vtu_in_parallel(filename, mpi_communicator);
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
        relevant_solution,
        estimated_error_per_cell,
        fe.component_mask(velocities));

    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        triangulation, estimated_error_per_cell, 0.3, 0.1);
    triangulation.execute_coarsening_and_refinement();
}

template <int dim>
BlockVector<double>
CommonCFD<dim>::get_solution()
{
    return relevant_solution;
}

template <int dim>
void
CommonCFD<dim>::write_timer_to_csv(const double total_time)
{
    if (mpi_rank != 0)
        return; // Only the root process writes the timer data

    namespace fs            = std::filesystem;
    std::string filename    = output_base_name + "timing.csv";
    const bool  file_exists = fs::exists(filename);

    std::ofstream           file(filename, std::ios::app);
    TimerOutput::OutputData data_type =
        TimerOutput::OutputData::total_wall_time;
    const auto timing_data = computing_timer.get_summary_data(data_type);

    if (!file_exists)
        {
            file << "MPI_Size,N_DOFs,total_time,";
            for (const auto &entry : timing_data)
                file << entry.first << ",";
            file << std::endl;
        }
    file << mpi_size << "," << dof_handler.n_dofs() << "," << total_time << ",";
    for (const auto &entry : timing_data)
        file << entry.second << ",";
    file << std::endl;
}
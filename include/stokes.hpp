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
class Stokes
{
  public:
    Stokes(unsigned int degree);
    void
    run();

  private:
    std::string         mesh_file = "../mesh.msh";
    unsigned int        degree    = 2;
    Triangulation<dim>  triangulation;
    const FESystem<dim> fe;
    DoFHandler<dim>     dof_handler;
    TimerOutput         computing_timer;

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
    void
    setup_system_matrix();
    void
    setup_preconditioner_matrix();
    void
    setup_system();

    void
    build_local_matrix(std::vector<SymmetricTensor<2, dim>> &symgrad_phi_u,
                       std::vector<double>                  &div_phi_u,
                       std::vector<Tensor<1, dim>>          &phi_u,
                       std::vector<double>                  &phi_p,
                       double                                JxW,
                       const unsigned int                    dofs_per_cell,
                       FullMatrix<double>                   &local_matrix);
    void
    build_local_preconditioner_matrix(
        std::vector<double> &phi_p,
        double               JxW,
        const unsigned int   dofs_per_cell,
        FullMatrix<double>  &local_preconditioner_matrix);
    void
    assemble_system();
    void
    solve();
    void
    output_results(unsigned int cycle);
    void
    refine_grid();
};

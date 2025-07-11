#include <deal.II/base/config.h>
#include <deal.II/base/function.h>

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

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include "../include/boundary.hpp"

using namespace dealii;

template <int dim> class Stokes {

public:
  Stokes(unsigned int degree);
  void run();

private:
  std::string mesh_file = "../mesh.msh";
  unsigned int degree = 2;
  Triangulation<dim> triangulation;
  const FESystem<dim> fe;
  DoFHandler<dim> dof_handler;
  AffineConstraints<double> constraints;

  BlockSparsityPattern sparsity_pattern;

  BlockVector<double> solution;
  BlockVector<double> system_rhs;
  BlockSparseMatrix<double> system_matrix;

  double viscosity = 0.1; // Oil viscosity

  void load_grid();

  void setup_system();

  void assemble_system();
  void solve();
  void output_results();
  void refine_grid();
};

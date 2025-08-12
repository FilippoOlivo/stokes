#pragma once
#include <deal.II/base/smartpointer.h>

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_bicgstab.h>

#include <deal.II/lac/trilinos_tpetra_precondition.h>
#include <deal.II/lac/trilinos_tpetra_solver_direct.h>
#include <deal.II/lac/trilinos_tpetra_vector.h>      

using namespace dealii;

template <class MatrixType>
class InverseMatrix : public Subscriptor
{
public:
  InverseMatrix(const MatrixType &m, bool use_direct_solver = false);

  void vmult(LinearAlgebra::TpetraWrappers::Vector<double> &      dst,
             const LinearAlgebra::TpetraWrappers::Vector<double> &src) const;

private:
  const SmartPointer<const MatrixType> matrix;
  SolverControl                        solver_control;
  bool                                 use_direct_solver;

  // Amesos2-based direct solver (Tpetra):
  mutable std::shared_ptr<LinearAlgebra::TpetraWrappers::SolverDirect<double>> solver_direct;

  // Iterative solver:
  mutable std::shared_ptr<SolverFGMRES<LinearAlgebra::TpetraWrappers::Vector<double>>> solver_gmres;

  // Ifpack2 ILUT (or switch to PreconditionAMG for MueLu):
  mutable LinearAlgebra::TpetraWrappers::PreconditionILUT<double> preconditioner;
};

template <class MatrixType>
InverseMatrix<MatrixType>::InverseMatrix(const MatrixType &m, bool use_direct)
  : matrix(&m)
  , solver_control(m.m(), 1e-6)   // max iters = rows, tol = 1e-6
  , use_direct_solver(use_direct)
{
  if (use_direct_solver)
  {
    LinearAlgebra::TpetraWrappers::SolverDirect<double>::AdditionalData data("SuperLU_DIST");
    solver_direct = std::make_shared<LinearAlgebra::TpetraWrappers::SolverDirect<double>>(solver_control, data);
    solver_direct->initialize(*matrix);
    return;
  }

  // Ifpack2 ILUT preconditioner
  LinearAlgebra::TpetraWrappers::PreconditionILUT<double>::AdditionalData ilut_params;
  // (optional) tune ILUT:
  // ilut_params.fill_level = 2.0;
  // ilut_params.drop_tolerance = 1e-4;

  preconditioner.initialize(*matrix, ilut_params);

  solver_gmres = std::make_shared<SolverFGMRES<LinearAlgebra::TpetraWrappers::Vector<double>>>(solver_control);
}

template <class MatrixType>
void InverseMatrix<MatrixType>::vmult(LinearAlgebra::TpetraWrappers::Vector<double> &      dst,
                                      const LinearAlgebra::TpetraWrappers::Vector<double> &src) const
{
  if (use_direct_solver)
  {
    solver_direct->solve(dst, src);
    return;
  }
  solver_gmres->solve(*matrix, dst, src, preconditioner);
}
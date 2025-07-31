#pragma once
#include <deal.II/base/smartpointer.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_vector.h>

using namespace dealii;

template <class MatrixType, class PreconditionerType>
class InverseMatrix : public Subscriptor
{
  public:
    InverseMatrix(const MatrixType         &m,
                  const PreconditionerType &preconditioner);

    void
    vmult(TrilinosWrappers::MPI::Vector       &dst,
          const TrilinosWrappers::MPI::Vector &src) const;

  private:
    const SmartPointer<const MatrixType>         matrix;
    const SmartPointer<const PreconditionerType> preconditioner;
};

template <class MatrixType, class PreconditionerType>
InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix(
    const MatrixType         &m,
    const PreconditionerType &preconditioner)
    : matrix(&m)
    , preconditioner(&preconditioner)
{}

template <class MatrixType, class PreconditionerType>
void
InverseMatrix<MatrixType, PreconditionerType>::vmult(
    TrilinosWrappers::MPI::Vector       &dst,
    const TrilinosWrappers::MPI::Vector &src) const
{
    SolverControl solver_control(src.size(), 1e-14 * src.l2_norm());
    TrilinosWrappers::SolverDirect::AdditionalData data;
    data.solver_type = "Amesos_Mumps"; // o "SuperLU", "MUMPS", etc.

    TrilinosWrappers::SolverDirect solver(solver_control, data);

    dst = 0;
    solver.solve(*matrix, dst, src);
}
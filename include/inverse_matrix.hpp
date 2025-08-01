#pragma once
#include <deal.II/base/smartpointer.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_vector.h>

using namespace dealii;

template <class MatrixType>
class InverseMatrix : public Subscriptor
{
  public:
    InverseMatrix(const MatrixType &m);

    void
    vmult(TrilinosWrappers::MPI::Vector       &dst,
          const TrilinosWrappers::MPI::Vector &src) const;

  private:
    const SmartPointer<const MatrixType>                    matrix;
    SolverControl                                           solver_control;
    mutable std::shared_ptr<TrilinosWrappers::SolverDirect> solver;
};

template <class MatrixType>
InverseMatrix<MatrixType>::InverseMatrix(const MatrixType &m)
    : matrix(&m)
    , solver_control(m.m(), 1e-12)
{
    TrilinosWrappers::SolverDirect::AdditionalData data;
    data.solver_type = "Amesos_Mumps";
    solver =
        std::make_shared<TrilinosWrappers::SolverDirect>(solver_control, data);

    solver->initialize(*matrix);
}

template <class MatrixType>
void
InverseMatrix<MatrixType>::vmult(TrilinosWrappers::MPI::Vector       &dst,
                                 const TrilinosWrappers::MPI::Vector &src) const
{
    dst = 0;
    solver->solve(dst, src);
}
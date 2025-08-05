#pragma once
#include <deal.II/base/smartpointer.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_vector.h>

using namespace dealii;

template <class MatrixType>
class InverseMatrix : public Subscriptor
{
  public:
    InverseMatrix(const MatrixType &m, bool use_direct_solver = false);

    void
    vmult(TrilinosWrappers::MPI::Vector       &dst,
          const TrilinosWrappers::MPI::Vector &src) const;

  private:
    const SmartPointer<const MatrixType>                    matrix;
    SolverControl                                           solver_control;
    bool                                                    use_direct_solver;
    mutable std::shared_ptr<TrilinosWrappers::SolverDirect> solver_direct;
};

template <class MatrixType>
InverseMatrix<MatrixType>::InverseMatrix(const MatrixType &m,
                                         bool              use_direct_solver)
    : matrix(&m)
    , solver_control(m.m(), 1e-12)
    , use_direct_solver(use_direct_solver)
{
    if (use_direct_solver)
        {
            // Use a direct solver
            TrilinosWrappers::SolverDirect::AdditionalData data;
            data.solver_type = "Amesos_Superludist";
            solver_direct =
                std::make_shared<TrilinosWrappers::SolverDirect>(solver_control,
                                                                 data);
            solver_direct->initialize(*matrix);
        }
}

template <class MatrixType>
void
InverseMatrix<MatrixType>::vmult(TrilinosWrappers::MPI::Vector       &dst,
                                 const TrilinosWrappers::MPI::Vector &src) const
{
    if (use_direct_solver)
        {
            // Use the direct solver
            solver_direct->solve(dst, src);
            return;
        }
    TrilinosWrappers::PreconditionAMG preconditioner;
    preconditioner.initialize(
        *matrix, TrilinosWrappers::PreconditionAMG::AdditionalData());
    SolverControl solver_control(matrix->m(), 1e-12);
    SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres(solver_control);
    solver_gmres.solve(*matrix, dst, src, preconditioner);
}
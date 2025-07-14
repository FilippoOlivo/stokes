#include <deal.II/base/smartpointer.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector.h>

using namespace dealii;

template <class MatrixType, class PreconditionerType>
class InverseMatrix : public Subscriptor {
public:
  InverseMatrix(const MatrixType &m, const PreconditionerType &preconditioner);

  void vmult(Vector<double> &dst, const Vector<double> &src) const;

private:
  const SmartPointer<const MatrixType> matrix;
  const SmartPointer<const PreconditionerType> preconditioner;
};

template <class MatrixType, class PreconditionerType>
InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix(
    const MatrixType &m, const PreconditionerType &preconditioner)
    : matrix(&m), preconditioner(&preconditioner) {}

template <class MatrixType, class PreconditionerType>
void InverseMatrix<MatrixType, PreconditionerType>::vmult(
    Vector<double> &dst, const Vector<double> &src) const {
  SolverControl solver_control(src.size(), 1e-6 * src.l2_norm());
  SolverCG<Vector<double>> cg(solver_control);

  dst = 0;

  cg.solve(*matrix, dst, src, *preconditioner);
}

#include <deal.II/base/config.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include "inverse_matrix.hpp"

using namespace dealii;

template <class PreconditionerType> class SchurComplement : public Subscriptor {
public:
  SchurComplement(
      const BlockSparseMatrix<double> &system_matrix,
      const InverseMatrix<SparseMatrix<double>, PreconditionerType> &A_inverse);

  void vmult(Vector<double> &dst, const Vector<double> &src) const;

private:
  const SmartPointer<const BlockSparseMatrix<double>> system_matrix;
  const SmartPointer<
      const InverseMatrix<SparseMatrix<double>, PreconditionerType>>
      A_inverse;

  mutable Vector<double> tmp1, tmp2;
};

template <class PreconditionerType>
SchurComplement<PreconditionerType>::SchurComplement(
    const BlockSparseMatrix<double> &system_matrix,
    const InverseMatrix<SparseMatrix<double>, PreconditionerType> &A_inverse)
    : system_matrix(&system_matrix), A_inverse(&A_inverse),
      tmp1(system_matrix.block(0, 0).m()), tmp2(system_matrix.block(0, 0).m()) {
}

template <class PreconditionerType>
void SchurComplement<PreconditionerType>::vmult(
    Vector<double> &dst, const Vector<double> &src) const {
  system_matrix->block(0, 1).vmult(tmp1, src);
  A_inverse->vmult(tmp2, tmp1);
  system_matrix->block(1, 0).vmult(dst, tmp2);
}
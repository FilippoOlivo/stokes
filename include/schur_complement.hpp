#include <deal.II/base/config.h>

#include <deal.II/base/smartpointer.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/numerics/vector_tools.h>

#include "inverse_matrix.hpp"

using namespace dealii;

// template <class PreconditionerType>
class SchurComplement : public Subscriptor
{
  public:
    SchurComplement(
        const TrilinosWrappers::BlockSparseMatrix           &system_matrix,
        const InverseMatrix<TrilinosWrappers::SparseMatrix> &A_inverse,
        const std::vector<IndexSet>                         &owned_partitioning,
        const MPI_Comm                                      &mpi_communicator);

    void
    vmult(TrilinosWrappers::MPI::Vector       &dst,
          const TrilinosWrappers::MPI::Vector &src) const;

  private:
    const SmartPointer<const TrilinosWrappers::BlockSparseMatrix> system_matrix;
    const SmartPointer<const InverseMatrix<TrilinosWrappers::SparseMatrix>>
        A_inverse;

    mutable TrilinosWrappers::MPI::Vector tmp1, tmp2;
    MPI_Comm                              mpi_communicator;
    std::vector<IndexSet>                 owned_partitioning;
};

// template <class PreconditionerType>
inline SchurComplement::SchurComplement(
    const TrilinosWrappers::BlockSparseMatrix           &system_matrix,
    const InverseMatrix<TrilinosWrappers::SparseMatrix> &A_inverse,
    const std::vector<IndexSet>                         &owned_partitioning,
    const MPI_Comm                                      &mpi_communicator)
    : system_matrix(&system_matrix)
    , A_inverse(&A_inverse)
    , tmp1(owned_partitioning[0], mpi_communicator)
    , tmp2(owned_partitioning[0], mpi_communicator)
    , mpi_communicator(mpi_communicator)
    , owned_partitioning(owned_partitioning)
{}

// template <class PreconditionerType>
inline void
SchurComplement::vmult(TrilinosWrappers::MPI::Vector       &dst,
                       const TrilinosWrappers::MPI::Vector &src) const
{
    system_matrix->block(0, 1).vmult(tmp1, src);
    A_inverse->vmult(tmp2, tmp1);
    system_matrix->block(1, 0).vmult(dst, tmp2);
    TrilinosWrappers::MPI::Vector tmp_mass(owned_partitioning[1],
                                           mpi_communicator);
    system_matrix->block(1, 1).vmult(tmp_mass, src);
    dst += tmp_mass;
}
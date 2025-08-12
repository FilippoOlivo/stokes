#pragma once
#include <deal.II/base/config.h>

#include <deal.II/base/smartpointer.h>

#include <deal.II/lac/trilinos_tpetra_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_tpetra_block_vector.h>
#include <deal.II/lac/trilinos_tpetra_vector.h>
#include <deal.II/lac/trilinos_tpetra_precondition.h>

#include <deal.II/numerics/vector_tools.h>

#include "inverse_matrix.hpp"

using namespace dealii;

template <class PreconditionerMp>
class BlockSchurPreconditioner : public Subscriptor
{
  public:
    BlockSchurPreconditioner(double viscosity,
                             const LinearAlgebra::TpetraWrappers::BlockSparseMatrix<double> &S,
                             const LinearAlgebra::TpetraWrappers::SparseMatrix<double> &P,
                             const PreconditionerMp &     Mppreconditioner,
                             const MPI_Comm &             mpi_communicator,
                             const std::vector<IndexSet> &owned_partitioning,
                             double                       gamma = 0.0,
                             bool use_direct_solver             = false);

    void
    vmult(LinearAlgebra::TpetraWrappers::BlockVector<double> &      dst,
          const LinearAlgebra::TpetraWrappers::BlockVector<double> &src) const;

  private:
    const double                               gamma;
    const double                               viscosity;
    const LinearAlgebra::TpetraWrappers::BlockSparseMatrix<double> &stokes_matrix;
    const LinearAlgebra::TpetraWrappers::SparseMatrix<double> &     pressure_mass_matrix;
    const PreconditionerMp &                   mp_preconditioner;
    const MPI_Comm &                           mpi_communicator;
    const std::vector<IndexSet> &              owned_partitioning;

    InverseMatrix<LinearAlgebra::TpetraWrappers::SparseMatrix<double>> A_inverse;
};

template <class PreconditionerMp>
BlockSchurPreconditioner<PreconditionerMp>::BlockSchurPreconditioner(
    double                                     viscosity,
    const LinearAlgebra::TpetraWrappers::BlockSparseMatrix<double> &S,
    const LinearAlgebra::TpetraWrappers::SparseMatrix<double> &     P,
    const PreconditionerMp &                   Mppreconditioner,
    const MPI_Comm &                           mpi_communicator,
    const std::vector<IndexSet> &              owned_partitioning,
    double                                     gamma,
    bool                                       use_direct_solver)
    : gamma(gamma)
    , viscosity(viscosity)
    , stokes_matrix(S)
    , pressure_mass_matrix(P)
    , mp_preconditioner(Mppreconditioner)
    , mpi_communicator(mpi_communicator)
    , owned_partitioning(owned_partitioning)
    , A_inverse(stokes_matrix.block(0, 0), use_direct_solver)
{}

template <class PreconditionerMp>
void
BlockSchurPreconditioner<PreconditionerMp>::vmult(
    LinearAlgebra::TpetraWrappers::BlockVector<double> &      dst,
    const LinearAlgebra::TpetraWrappers::BlockVector<double> &src) const
{
    LinearAlgebra::TpetraWrappers::Vector<double> utmp(owned_partitioning[0], mpi_communicator);

    SolverControl solver_control(1000, 1e-6 * src.block(1).l2_norm());
    SolverCG<LinearAlgebra::TpetraWrappers::Vector<double>> cg(solver_control);

    dst.block(1) = 0.0;
    cg.solve(pressure_mass_matrix,
             dst.block(1),
             src.block(1),
             mp_preconditioner);
    dst.block(1) *= -(viscosity + gamma);

    stokes_matrix.block(0, 1).vmult(utmp, dst.block(1));
    utmp *= -1.0;
    utmp += src.block(0);

    A_inverse.vmult(dst.block(0), utmp);
    LinearAlgebra::TpetraWrappers::Vector<double> tmp_mass(owned_partitioning[1],
                                           mpi_communicator);
    stokes_matrix.block(1, 1).vmult(tmp_mass, src.block(1));
    dst.block(1) += tmp_mass;
}

#pragma once
#include <deal.II/base/config.h>

#include <deal.II/base/smartpointer.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/numerics/vector_tools.h>

#include "inverse_matrix.hpp"

using namespace dealii;

template <class PreconditionerMp>
class BlockSchurPreconditioner : public Subscriptor
{
  public:
    BlockSchurPreconditioner(double viscosity,
                             const TrilinosWrappers::BlockSparseMatrix &S,
                             const TrilinosWrappers::SparseMatrix &     P,
                             const PreconditionerMp &     Mppreconditioner,
                             const MPI_Comm &             mpi_communicator,
                             const std::vector<IndexSet> &owned_partitioning,
                             double                       gamma = 0.0,
                             bool use_direct_solver             = false);

    void
    vmult(TrilinosWrappers::MPI::BlockVector &      dst,
          const TrilinosWrappers::MPI::BlockVector &src) const;

  private:
    const double                               gamma;
    const double                               viscosity;
    const TrilinosWrappers::BlockSparseMatrix &stokes_matrix;
    const TrilinosWrappers::SparseMatrix &     pressure_mass_matrix;
    const PreconditionerMp &                   mp_preconditioner;
    const MPI_Comm &                           mpi_communicator;
    const std::vector<IndexSet> &              owned_partitioning;

    InverseMatrix<TrilinosWrappers::SparseMatrix> A_inverse;
};

template <class PreconditionerMp>
BlockSchurPreconditioner<PreconditionerMp>::BlockSchurPreconditioner(
    double                                     viscosity,
    const TrilinosWrappers::BlockSparseMatrix &S,
    const TrilinosWrappers::SparseMatrix &     P,
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
    TrilinosWrappers::MPI::BlockVector &      dst,
    const TrilinosWrappers::MPI::BlockVector &src) const
{
    TrilinosWrappers::MPI::Vector utmp(owned_partitioning[0], mpi_communicator);

    SolverControl solver_control(1000, 1e-6 * src.block(1).l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> cg(solver_control);

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
    TrilinosWrappers::MPI::Vector tmp_mass(owned_partitioning[1],
                                           mpi_communicator);
    stokes_matrix.block(1, 1).vmult(tmp_mass, src.block(1));
    dst.block(1) += tmp_mass;
}

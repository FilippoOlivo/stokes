#include "base_stokes.hpp"

template <int dim>
class StableStokes : public BaseStokes<dim>
{
  public:
    StableStokes(const Parameters &params)
        : BaseStokes<dim>(params, "stable_stokes-")
    {}

  private:
    void
    setup_system_matrix() override;
    void
    build_local_matrix(std::vector<SymmetricTensor<2, dim>> &symgrad_phi_u,
                       std::vector<double>                  &div_phi_u,
                       std::vector<double>                  &phi_p,
                       double                                JxW,
                       const unsigned int                    dofs_per_cell,
                       FullMatrix<double>                   &local_matrix);
    void
    assemble_system() override;
};
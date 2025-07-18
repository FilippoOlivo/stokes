#include "base_stokes.hpp"

template <int dim>
class StabilizedStokes : public BaseStokes<dim>
{
  public:
    StabilizedStokes(unsigned int degree, std::string mesh_file)
        : BaseStokes<dim>(degree, degree, mesh_file, "stabilized_stokes-")
    {}

  private:
    double delta = 1e-1; // Stabilization parameter
    void
    setup_system_matrix() override;

    void
    build_local_matrix(std::vector<SymmetricTensor<2, dim>> &symgrad_phi_u,
                       std::vector<double>                  &div_phi_u,
                       std::vector<Tensor<1, dim>>          &grad_phi_p,
                       std::vector<double>                  &phi_p,
                       double                                JxW,
                       const unsigned int                    dofs_per_cell,
                       FullMatrix<double>                   &local_matrix,
                       double                                h_k_squared);
    void
    assemble_system() override;
};
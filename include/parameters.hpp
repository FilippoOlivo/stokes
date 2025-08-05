#pragma once
#include <deal.II/base/config.h>

#include <deal.II/base/parameter_acceptor.h>

class Parameters : public dealii::ParameterAcceptor
{
  public:
    Parameters()
        : ParameterAcceptor("Problem Parmeters")
    {
        add_parameter("mesh_file", mesh_file, "File where to find the mesh");
        add_parameter("viscosity", viscosity, "Viscosity of the fluid");
        add_parameter("inlet_velocity", inlet_velocity, "Velocity at inlet");
        add_parameter("degree_p", degree_p, "FEM degree");
        add_parameter("degree_u", degree_u, "FEM degree");
        add_parameter("refinement_steps", refinement_steps, "Refinement steps");
    }
    std::string  mesh_file        = "mesh.msh";
    double       viscosity        = 1.0;
    double       inlet_velocity   = 1.0;
    unsigned int degree_p         = 1;
    unsigned int degree_u         = 2;
    unsigned int refinement_steps = 5;
};

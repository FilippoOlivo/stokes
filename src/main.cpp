#include "../include/navier_stokes.hpp"

int
main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    Parameters                       params;
    ParameterAcceptor::initialize("../parameters.prm");
    std::string mesh_file;
    if (argc == 2)
        mesh_file = argv[1];
    else
        mesh_file = "../mesh_0.msh";

    NavierStokes<2> stokes(params);
    stokes.run();
}
#include "../include/navier_stokes.hpp"

int
main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    Parameters                       params;

    std::string parameter_file;
    ;
    if (argc == 2)
        parameter_file = argv[1];
    else
        parameter_file = "parameters.prm";
    ParameterAcceptor::initialize(parameter_file);
    NavierStokes<2> stokes(params);
    stokes.run();
}
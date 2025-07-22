#include "../include/stable_stokes.hpp"


int
main(int argc, char *argv[])
{
    Parameters params;
    ParameterAcceptor::initialize("../parameters.prm");
    std::string mesh_file;
    if (argc == 2)
        mesh_file = argv[1];
    else
        mesh_file = "../mesh_0.msh";

    StableStokes<2> stokes(params);
    stokes.run();
}
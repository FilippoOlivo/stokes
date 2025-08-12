#include "../include/navier_stokes.hpp"
// #include "../include/stabilized_stokes.hpp"
// #include "../include/stable_stokes.hpp"

int
main(int argc, char *argv[])
{   
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    Parameters                       params;

    std::string parameter_file;
    if (argc == 2)
        parameter_file = argv[1];
    else
        parameter_file = "parameters.prm";
    ParameterAcceptor::initialize(parameter_file);
    std::string model_name = "NavierStokes";
    if (argc == 3)
        model_name = argv[2];
    // if (model_name == "StableStokes")
    //     {
    //         StableStokes<2> stable_stokes(params);
    //         stable_stokes.run();
    //         return 0;
    //     }
    // else if (model_name == "StabilizedStokes")
    //     {
    //         StabilizedStokes<2> stabilized_stokes(params);
    //         stabilized_stokes.run();
    //         return 0;
    //     }
    // if (model_name == "NavierStokes")
    //     {
            NavierStokes<2> navier_stokes(params);
            navier_stokes.run();
            std::cout << "Navier-Stokes simulation completed." << std::endl;
            return 0;
        // }
    // else
    //     {
    //         std::cerr << "Unknown model name: " << model_name
    //                   << ". Supported models are: NavierStokes, StableStokes, "
    //                      "StabilizedStokes."
    //                   << std::endl;
    //         return 1;
    //     }
}
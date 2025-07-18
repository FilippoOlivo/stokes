#include "../include/stable_stokes.hpp"


int
main()
{
    StableStokes<2> stokes(1, "../mesh_0.msh");
    stokes.run();
}
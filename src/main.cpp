#include "../include/BP_stokes.hpp"


int
main()
{
    StabilizedStokes<2> stokes(1);
    stokes.run();
}
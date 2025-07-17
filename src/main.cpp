#include "../include/stokes.hpp"

int
main()
{
    Stokes<2> stokes(1);
    stokes.run();
    return 0;
}
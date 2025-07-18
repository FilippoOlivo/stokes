#include "../include/boundary.hpp"

double
InletBoundary::value(const Point<2> &p, const unsigned int component) const
{
    if (component == 0)
        if (p[0] <= 1e-8)
            return 4 * p[1] * (1 - p[1]);
    return 0;
}

void
InletBoundary::vector_value(const Point<2> &p, Vector<double> &values) const
{
    for (unsigned int c = 0; c < this->n_components; ++c)
        values(c) = InletBoundary::value(p, c);
}

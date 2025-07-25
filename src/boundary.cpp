#include "../include/boundary.hpp"

double
InletBoundary::value(const Point<2> &p, const unsigned int component) const
{
    if (component == 0)
        if (p[0] <= 1e-8)
            return 4 * inlet_velocity * p[1] * (1.5 - p[1]) / (1.5 * 1.5);
    return 0;
}

void
InletBoundary::vector_value(const Point<2> &p, Vector<double> &values) const
{
    for (unsigned int c = 0; c < this->n_components; ++c)
        values(c) = InletBoundary::value(p, c);
}

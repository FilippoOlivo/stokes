#include <deal.II/base/config.h>

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>

using namespace dealii;

class InletBoundary : public Function<2>
{
  public:
    InletBoundary(double inlet_velocity = 1.0)
        : Function<2>(3)
        , inlet_velocity(inlet_velocity)
    {
        Assert(inlet_velocity >= 0,
               ExcMessage("Inlet velocity must be non-negative."));
    }

    virtual double
    value(const Point<2> &p, const unsigned int component = 0) const override;

    virtual void
    vector_value(const Point<2> &p, Vector<double> &value) const override;

  private:
    double inlet_velocity;
};

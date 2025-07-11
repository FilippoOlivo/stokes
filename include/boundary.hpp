#include <deal.II/base/config.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>

using namespace dealii;

class InletBoundary : public Function<2> {
public:
  InletBoundary() : Function<2>(3) {}

  virtual double value(const Point<2> &p,
                       const unsigned int component = 0) const override;

  virtual void vector_value(const Point<2> &p,
                            Vector<double> &value) const override;
};
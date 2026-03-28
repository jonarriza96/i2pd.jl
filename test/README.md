# Tests for i2pd.jl

We group the tests as follows:

## QP solution

- **Description**:
  - Solve QP and validate solutions
- **Settings**:
  - sparse, dense
  - BACKEND_DIRECT, BACKEND_ITERATIVE
  - MODE_UNCONDENSED, MODE_CONDENSED
  - MODE_UNCONDENSED_IMPLICIT, MODE_CONDENSED_IMPLICIT
- **Problems**:
  - simple (dummy) QP
  - Maros Meszaros

## Ruiz Scaling

- **Description**:
  - Check that norms of rows and columns are "scaled"
  - Scale and Unscale solution
- **Settings**:
  - MODE_UNCONDENSED, MODE_CONDENSED
  - MODE_UNCONDENSED_IMPLICIT, MODE_CONDENSED_IMPLICIT
- **Problems**:
  - simple (dummy) QP
  - Maros Meszaros

# The certification interface

Every query in this package answers a question about an arbitrary Julia function, and by
Rice's theorem no such semantic question is decidable in general. The package therefore
commits to a single, uniform contract, chosen so that a wrong answer can never be the unsafe
one:

## One-sided certificates

For the `is*` family (`islinear`, `isquadratic`, `isautonomous`, `issmooth`, `ispure`,
`isinferable`), **`true` is a proof** of the property along every execution path, and `false`
only means *not proven* — the function may still have the property (for example
`x^2 - x^2 + x` is linear but is not certified, because the degree bound does not model
cancellation). Dispatch on `true`; treat `false` as "use the general path."

For the `has*` family (`hasbranching`, `hasrandomness`, `hasmutation`), the polarity flips so
that the same rule holds: **`false` is the certificate** (proven absent), and `true` means the
property was observed *or could not be ruled out*.

**Every internal "give up" — an unanalyzable call, an exhausted recursion budget, a foreign
number type, an aborted trace — resolves to the conservative answer**, never to the
certificate.

## Dual guards

Two complementary engines back the certificates, because each covers the other's blind spot:

  - a **static scan of the type-inferred IR** (the `hasbranching` machinery), which sees every
    execution path but deliberately treats `Base`/standard-library internals as opaque leaves;
  - **dynamic tracer types** (`islinear`'s degree tracer, `issmooth`'s smoothness probe,
    `hasmutation`'s write probe), which follow values through any generic code — including
    library internals such as broadcasting, where the static scan is leaf-blind — but only
    along the path actually taken. Every operation on a tracer that would need the traced
    *value* (a comparison, `isnan`, rounding, conversion) throws, so a computation that the
    tracer cannot faithfully follow aborts toward the conservative answer instead of producing
    an unsound certificate.

A tracer-based certificate is only issued when `hasbranching` additionally proves the traced
path is the *only* path.

## Independent validation

The test suite checks certificates against mathematics, not against the implementation's own
rules: linear certificates are cross-validated with exactly vanishing finite differences over
`Rational{BigInt}` arithmetic at random points, and non-mutation certificates against exact
before/after comparison of real calls. Each property is exercised by adversarial cases
(recursion towers, wrapper indirection, cancellation, aliasing, exotic number types), and
contributions of new properties are expected to bring an independent refutation oracle and an
adversarial battery of their own.

## Known boundaries

The certificates share the documented visibility limits of the static scan: calls hidden
behind dynamic dispatch (untyped `Function` fields) are invisible to it, `Base` internals are
leaves for branch detection (tracers compensate where they run), and floating-point rounding
is not modeled — polynomial and smoothness certificates are statements about the program's
real-arithmetic semantics.

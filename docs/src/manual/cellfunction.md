# Cell function

!!! warning
    This page is under construction.

As explained earlier, at least two coordinates systems exist in Bcube : the "reference" coordinates (`ReferenceDomain`) and the "physical" coordinates (`PhysicalDomain`). The evaluation of a function on a point in a cell depends on the way this point has been defined. Hence the definition of `CellPoint`s that embed the coordinate system. Given a `CellPoint` (or eventually a `FacePoint`), an `AbstractCellFunction` will be evaluated and the mapping between the `ReferenceDomain` to the `PhysicalDomain` (or reciprocally) will be performed internally if necessary : if an `AbstractCellFunction` defined in terms of reference coordinates is applied on a `CellPoint` expressed in the reference coordinates system, no mapping is needed.

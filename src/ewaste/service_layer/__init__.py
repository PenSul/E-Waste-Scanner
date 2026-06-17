"""Service layer: use-case orchestration.

The service layer wires a detector, a market-price provider, and the reference
repositories together to turn an uploaded image into a fully costed scan result
and a digital receipt. It contains no business rules itself -- those live in
:mod:`ewaste.domain` -- only coordination.
"""

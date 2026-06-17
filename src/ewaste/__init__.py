"""E-Waste Scanner.

An upload-based application that detects electronic waste in a photograph and
estimates its recoverable material value and environmental impact.

The package follows the ports-and-adapters layering from *Architecture Patterns
with Python* (Percival & Gregory):

* :mod:`ewaste.domain` -- pure domain model and calculations (no I/O).
* :mod:`ewaste.adapters` -- concrete implementations of I/O ports
  (object detector, market-price provider, reference-data repositories).
* :mod:`ewaste.service_layer` -- orchestration of a single scan and the
  building of the digital receipt.

The Streamlit user interface in ``app/`` is a thin entrypoint that depends only
on the service layer.
"""

__version__ = "0.1.0"

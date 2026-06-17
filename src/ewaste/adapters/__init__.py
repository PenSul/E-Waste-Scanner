"""Adapters layer: concrete implementations of the domain ports.

Each adapter wraps an external dependency behind an abstract port declared in
:mod:`ewaste.ports`, so the domain and service layers stay testable in
isolation: the YOLO object detector, the yfinance market-price provider (with
caching and a static fallback), and the reference-data repositories.
"""

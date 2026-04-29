"""
src/__init__.py — Source Package Marker
========================================

Makes the ``src/`` directory a proper Python package so that modules
(fetcher, features, model, signal_engine, notifier, scheduler) can be
imported with:

    from src.fetcher import DataFetcher
    from src.model   import SignalModel
    ...
"""

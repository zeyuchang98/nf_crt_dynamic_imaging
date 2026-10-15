"""Compatibility entry point for the shared flat results index.

The CSV labels the evaluation protocol explicitly. Block and midpoint scores are
kept separate; use evaluation.render to request figures for selected runs.
"""
from .summarize import main


if __name__ == '__main__':
    main()

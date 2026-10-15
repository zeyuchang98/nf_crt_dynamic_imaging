"""Regenerate the flat results/summary.csv index from saved run records."""
import argparse
from pathlib import Path

from ..run_records import update_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    args = parser.parse_args()
    update_summary(args.results)
    print(args.results / 'summary.csv')


if __name__ == '__main__':
    main()

"""
Write a human genetic map from stdpopsim as a HapMap file, which tsinfer's
recombination_rate and run.py's --recombination-rate read.

stdpopsim is not a dependency of the benchmarks, so run this with it added:

    uv run --with stdpopsim python genetic_map.py \
        data/genetic_maps/chr20_HapMapII_GRCh38.txt

stdpopsim downloads and caches the maps for all chromosomes on first use.
"""

import pathlib

import click
import numpy as np
import tsls


def hapmap_rows(rate_map):
    """
    Return the (position, rate cM/Mb, map cM) rows of a HapMap file giving
    the specified msprime RateMap (rates per bp per generation). Intervals
    with missing (NaN) rates have no recombination: a HapMap file can't
    express missing data, and a leading missing interval is written by
    starting the map at its end with 0 cM, as msprime reads it.
    """
    left = rate_map.left
    rate = rate_map.rate
    cumulative_cm = rate_map.get_cumulative_mass(left) * 100
    rows = []
    for j in range(len(left)):
        if np.isnan(rate[j]):
            if j == 0:
                continue
            rows.append((int(left[j]), 0.0, float(cumulative_cm[j])))
        else:
            rows.append((int(left[j]), float(rate[j] * 1e8), float(cumulative_cm[j])))
    final_cm = float(rate_map.get_cumulative_mass([rate_map.sequence_length])[0] * 100)
    rows.append((int(rate_map.sequence_length), 0.0, final_cm))
    return rows


def write_hapmap(path, chromosome, rows):
    with open(path, "w") as f:
        print("Chromosome Position(bp) Rate(cM/Mb) Map(cM)", file=f)
        for position, rate, cm in rows:
            print(f"{chromosome} {position} {rate!r} {cm!r}", file=f)


@click.command()
@click.argument("output", type=click.Path(dir_okay=False))
@click.option("--chromosome", default="chr20", show_default=True)
@click.option("--genetic-map", default="HapMapII_GRCh38", show_default=True)
def main(output, chromosome, genetic_map):
    """
    Write the stdpopsim HomSap GENETIC_MAP for CHROMOSOME to OUTPUT in
    HapMap format, and check that tsls.model.read_genetic_map reads it back.
    """
    import stdpopsim  # optional dependency

    species = stdpopsim.get_species("HomSap")
    contig = species.get_contig(chromosome, genetic_map=genetic_map)
    rate_map = contig.recombination_map
    output = pathlib.Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_hapmap(output, chromosome, hapmap_rows(rate_map))

    positions = np.linspace(0, rate_map.sequence_length, 100_001)
    np.testing.assert_allclose(
        tsls.model.read_genetic_map(output, positions),
        rate_map.get_cumulative_mass(positions),
        rtol=1e-9,
        atol=1e-12,
    )
    click.echo(
        f"Wrote {rate_map.num_intervals} intervals to {output}: "
        f"{rate_map.total_mass:.4g} Morgans over {rate_map.sequence_length:.0f} bp"
    )


if __name__ == "__main__":
    main()

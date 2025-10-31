import logging
import sys
import pandas as pd
import numpy as np
from pybedtools import BedTool

# ------------------------------
# Logging setup
# ------------------------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger("SC-prepare")

# ------------------------------
# Main function
# ------------------------------
def main(mutation_file, element_file, callable_bed, output_file):
    """
    Process mutation and genomic element files to calculate:
      - total mutation counts per element (nMut)
      - number of unique samples per element (nSample)
      - element effective length (after intersection with callable regions)
      - total number of samples (N)
    """

    # --- Load input data ---
    logger.info("Loading input files...")
    mut = BedTool(mutation_file)              # e.g., chr10 1234547 1234548 C T DO34567
    elements = BedTool(element_file).sort()   # e.g., chr14 100002667 100005669 binID
    callable_regions = BedTool(callable_bed).sort()  # e.g., chr3 176752 186753

    # --- Compute total sample count ---
    N = len(pd.read_table(mutation_file, header=None, usecols=(5,))[5].unique())
    logger.info(f"Total unique samples detected: {N}")

    # --- Filter elements by callable regions ---
    old_cov = elements.total_coverage()
    filtered_elements = elements.intersect(callable_regions).sort()
    new_cov = filtered_elements.total_coverage()
    logger.info(
        "Whitelisted elements: {}/{} bp ({:.2f}%)".format(
            new_cov, old_cov, new_cov / old_cov * 100
        )
    )

    # --- Intersect mutations with elements ---
    cnames = [
        "chrom", "start", "end", "ref", "alt", "donor",
        "chrom2", "start2", "end2", "binID"
    ]
    mut_ele = mut.intersect(filtered_elements, wa=True, wb=True).to_dataframe(names=cnames)

    # --- Mutation counts per element ---
    logger.info("Calculating mutation statistics per element...")
    response_tab = mut_ele.pivot_table(
        index="binID", values="donor",
        aggfunc=[len, lambda x: len(x.unique())]
    )
    response_tab.columns = ["nMut", "nSample"]

    # --- Compute effective length of each element ---
    ele_df = filtered_elements.to_dataframe(names=["chrom", "start", "end", "binID"])
    ele_df["length"] = ele_df["end"] - ele_df["start"]
    eff_length = ele_df.pivot_table(index="binID", values="length", aggfunc=sum)

    # --- Combine results ---
    response_tab = eff_length.join(response_tab)
    response_tab = response_tab.fillna(0).astype(int)
    response_tab["N"] = N

    # --- Save output ---
    logger.info(f"Saving results to {output_file}")
    response_tab.to_csv(output_file, sep="\t")
    logger.info("✅ Finished successfully.")


# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python prepare.py <mutation_file> <element_file> <callable_bed> <output_file>")
        sys.exit(1)

    mutation_file, element_file, callable_bed, output_file = sys.argv[1:5]
    main(mutation_file, element_file, callable_bed, output_file)

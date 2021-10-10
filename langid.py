#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import argparse
import logging
import math
from pathlib import Path
import pdb

from probs import LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        nargs=2,
        help="path to the trained model"
    )
    parser.add_argument(
        "prior",
        type=float,
        help="Prior probability of genuen"
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0
    for (x, y, z) in read_trigrams(file, lm.vocab):
        prob = lm.prob(x, y, z)  # p(z | xy)
        log_prob += math.log(prob)
    return log_prob


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    log.info("Testing...")
    lm1 = LanguageModel.load(args.model[0])
    lm2 = LanguageModel.load(args.model[1])
    #assert lm1.vocab == lm2.vocab, "Make sure both models have the same vocabulary"
    prior1 = args.prior
    prior2 = 1 - prior1
    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # But we'd like to print a value in bits: so we convert
    # log base e to log base 2 at print time, by dividing by log(2).

    log.info("Per-file log-probabilities:")
    class1 = 'EN'
    class2 = 'SP'
    count_sp = 0
    count_en = 0
    for file in args.test_files:
        log_prob1: float = file_log_prob(file, lm1)
        log_prob2: float = file_log_prob(file, lm2)
        if log_prob1+math.log(prior1) > log_prob2+math.log(prior2):
            print(f"EN\t{file}")
            count_en+=1
        else:
            print(f"SP\t{file}")
            count_sp+=1
    per_sp = round(count_sp/(count_sp+count_en),2)*100
    per_en = round(count_en/(count_sp+count_en),2)*100
    print(f"{count_en} files were more probably English ({per_en}%)")
    print(f"{count_sp} files were more probably Spanish ({per_sp}%)")


if __name__ == "__main__":
    main()

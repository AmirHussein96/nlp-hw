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
import random

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to the trained model",
    )

    parser.add_argument(
        "n_samples",
        type=int,
    )
    parser.add_argument(
        "--max_length",
        type=int,
    )
    

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()

# def get_sample(lm, max_length):

#     lm.sample(max_length)
#     return lm.gen_sen

def get_sample(lm, max_length=20, start_symbol='BOS', end_symbol='EOS'):
    #     """ implementation od sampling method Q6
    #      Args:
                               
    #         max_length (int): max number of words in a generated sentence
        
    #     Returns:
    #         gen_sen: the generated random sentence 
        
    #     """
        
        gen_sen = ""
        x , y = "BOS", "BOS"
        return new_sample(lm, (x,y), gen_sen, max_length)
         
   
def new_sample(lm, context, gen_sen, remaining_expansions):
    remaining_expansions -= 1
    probs = []
    choice_opt = []
    x,y = context
    for z in lm.vocab:
        probs.append(lm.prob(x,y,z))
        choice_opt.append(z)
    sample =  random.choices(choice_opt, weights=probs, k=1)[0]
    gen_sen = gen_sen + " " + sample
    if len(gen_sen) > 0 and gen_sen[-1] == "EOS" or gen_sen[-3:] =="...":
        
        return gen_sen
        #gen_sen = gen_sen + "(" + node.name + " "
    elif remaining_expansions == 0 and gen_sen[-1] != "EOS":
        
        gen_sen += " ..."
        return gen_sen
    else:
        x, y = y, z 
    
    return new_sample(lm, (x,y), gen_sen, remaining_expansions)


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

    #log.info("Genrating text...")
    lm = LanguageModel.load(args.model)
    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # But we'd like to print a value in bits: so we convert
    # log base e to log base 2 at print time, by dividing by log(2).
   # log.info("Per-file log-probabilities:")
    tot_sen = ""
    #pdb.set_trace()
    for i in range(args.n_samples):

        gen_text = get_sample(lm, args.max_length)
        tot_sen =  tot_sen + " " + gen_text + "\n"
     #   log.info(gen_text)
    print(tot_sen)


if __name__ == "__main__":
    main()

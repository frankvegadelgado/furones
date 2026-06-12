#                   Approximate Minimum Dominating Set Solver
#                          Frank Vega
#                      February 5th, 2025

import argparse
import math
import time

from . import algorithm
from . import applogger
from . import parser
from . import utils


def approximate_solution(
    inputFile,
    verbose=False,
    log=False,
    count=False,
    bruteForce=False,
    approximation=False,
):
    """Find an approximate dominating set for one DIMACS graph."""
    logger = applogger.Logger(
        applogger.FileLogger() if log else applogger.ConsoleLogger(verbose)
    )

    logger.info("Parsing the Input File started")
    started = time.time()
    graph = parser.read(inputFile)
    filename = utils.get_file_name(inputFile)
    logger.info(f"Parsing the Input File done in: {(time.time() - started) * 1000.0} milliseconds")

    approximate_result = None
    brute_force_result = None

    if approximation:
        logger.info("An Approximate Solution with a logarithmic approximation ratio started")
        started = time.time()
        approximate_result = algorithm.find_dominating_set_approximation(graph)
        logger.info(
            "An Approximate Solution with a logarithmic approximation ratio "
            f"done in: {(time.time() - started) * 1000.0} milliseconds"
        )
        answer = utils.string_result_format(approximate_result, count)
        utils.println(f"{filename}: (Approximation) {answer}", logger, log)

    if bruteForce:
        logger.info("A solution with an exponential-time complexity started")
        started = time.time()
        brute_force_result = algorithm.find_dominating_set_brute_force(graph)
        logger.info(
            "A solution with an exponential-time complexity "
            f"done in: {(time.time() - started) * 1000.0} milliseconds"
        )
        answer = utils.string_result_format(brute_force_result, count)
        utils.println(f"{filename}: (Brute Force) {answer}", logger, log)

    logger.info("Furones Algorithm with a validated approximation solution started")
    started = time.time()
    novel_result = algorithm.find_dominating_set(graph)
    logger.info(
        "Furones Algorithm with a validated approximation solution "
        f"done in: {(time.time() - started) * 1000.0} milliseconds"
    )

    answer = utils.string_result_format(novel_result, count)
    utils.println(f"{filename}: {answer}", logger, log)

    if novel_result and (bruteForce or approximation):
        if bruteForce and brute_force_result:
            output = f"Exact Ratio (Furones/Optimal): {len(novel_result) / len(brute_force_result)}"
        elif approximation and approximate_result:
            output = (
                "Upper Bound for Ratio (Furones/Optimal): "
                f"{math.log2(graph.number_of_nodes()) * len(novel_result) / len(approximate_result)}"
            )
        else:
            output = None
        if output:
            utils.println(output, logger, log)


def main():
    helper = argparse.ArgumentParser(
        prog="asia",
        description="Solve the Approximate Minimum Dominating Set for an undirected graph encoded in DIMACS format.",
    )
    helper.add_argument("-i", "--inputFile", type=str, help="input file path", required=True)
    helper.add_argument(
        "-a",
        "--approximation",
        action="store_true",
        help="enable comparison with a polynomial-time approximation approach within a logarithmic factor",
    )
    helper.add_argument(
        "-b",
        "--bruteForce",
        action="store_true",
        help="enable comparison with the exponential-time brute-force approach",
    )
    helper.add_argument("-c", "--count", action="store_true", help="calculate the size of the Dominating Set")
    helper.add_argument("-v", "--verbose", action="store_true", help="enable verbose output")
    helper.add_argument("-l", "--log", action="store_true", help="enable file logging")
    helper.add_argument("--version", action="version", version="%(prog)s 0.2.9")

    args = helper.parse_args()
    approximate_solution(
        args.inputFile,
        verbose=args.verbose,
        log=args.log,
        count=args.count,
        bruteForce=args.bruteForce,
        approximation=args.approximation,
    )


if __name__ == "__main__":
    main()

# Created on 02/05/2025
# Author: Frank Vega

import argparse
import math
import time

from . import algorithm
from . import applogger
from . import parser
from . import utils


def restricted_float(x):
    try:
        value = float(x)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{x!r} is not a floating-point literal") from exc

    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError(f"{x!r} is not in range [0.0, 1.0]")
    return value


def main():
    helper = argparse.ArgumentParser(
        prog="test_asia",
        description="The Furones Testing Application using randomly generated sparse matrices.",
    )
    helper.add_argument("-d", "--dimension", type=int, help="matrix dimension", required=True)
    helper.add_argument("-n", "--num_tests", type=int, default=5, help="number of tests to run")
    helper.add_argument(
        "-s",
        "--sparsity",
        type=restricted_float,
        default=0.95,
        help="sparsity of the matrices (0.0 for dense, close to 1.0 for very sparse)",
    )
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
    helper.add_argument("-w", "--write", action="store_true", help="write generated random matrices to files")
    helper.add_argument("-v", "--verbose", action="store_true", help="enable verbose output")
    helper.add_argument("-l", "--log", action="store_true", help="enable file logging")
    helper.add_argument("--version", action="version", version="%(prog)s 0.2.9")

    args = helper.parse_args()
    logger = applogger.Logger(applogger.FileLogger() if args.log else applogger.ConsoleLogger(args.verbose))
    hash_string = utils.generate_short_hash(6 + math.ceil(math.log2(args.num_tests))) if args.write else None

    for i in range(args.num_tests):
        logger.info(f"Creating Matrix {i + 1}")
        sparse_matrix = utils.random_matrix_tests((args.dimension, args.dimension), args.sparsity)
        if sparse_matrix is None:
            continue

        graph = utils.sparse_matrix_to_graph(sparse_matrix)
        logger.info(f"Matrix shape: {sparse_matrix.shape}")
        logger.info(f"Number of non-zero elements: {sparse_matrix.nnz}")
        logger.info(f"Sparsity: {1 - (sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]))}")

        approximate_result = None
        brute_force_result = None

        if args.approximation:
            logger.info("An Approximate Solution with a logarithmic approximation ratio started")
            started = time.time()
            approximate_result = algorithm.find_dominating_set_approximation(graph)
            logger.info(
                "An Approximate Solution with a logarithmic approximation ratio "
                f"done in: {(time.time() - started) * 1000.0} milliseconds"
            )
            answer = utils.string_result_format(approximate_result, args.count)
            utils.println(f"{i + 1}-Approximation Test: {answer}", logger, args.log)

        if args.bruteForce:
            logger.info("A solution with an exponential-time complexity started")
            started = time.time()
            brute_force_result = algorithm.find_dominating_set_brute_force(graph)
            logger.info(
                "A solution with an exponential-time complexity "
                f"done in: {(time.time() - started) * 1000.0} milliseconds"
            )
            answer = utils.string_result_format(brute_force_result, args.count)
            utils.println(f"{i + 1}-Brute Force Test: {answer}", logger, args.log)

        logger.info("Furones Algorithm with a validated approximation solution started")
        started = time.time()
        novel_result = algorithm.find_dominating_set(graph)
        logger.info(
            "Furones Algorithm with a validated approximation solution "
            f"done in: {(time.time() - started) * 1000.0} milliseconds"
        )

        answer = utils.string_result_format(novel_result, args.count)
        utils.println(f"{i + 1}-Furones Test: {answer}", logger, args.log)

        if novel_result and (args.bruteForce or args.approximation):
            if args.bruteForce and brute_force_result:
                output = f"Exact Ratio (Furones/Optimal): {len(novel_result) / len(brute_force_result)}"
            elif args.approximation and approximate_result:
                output = (
                    "Upper Bound for Ratio (Furones/Optimal): "
                    f"{math.log2(graph.number_of_nodes()) * len(novel_result) / len(approximate_result)}"
                )
            else:
                output = None
            if output:
                utils.println(output, logger, args.log)

        if args.write:
            utils.println(f"Saving Matrix Test {i + 1}", logger, args.log)
            filename = f"sparse_matrix_{i + 1}_{hash_string}"
            parser.save_sparse_matrix_to_file(sparse_matrix, filename)
            utils.println(f"Matrix Test {i + 1} written to file {filename}.", logger, args.log)


if __name__ == "__main__":
    main()

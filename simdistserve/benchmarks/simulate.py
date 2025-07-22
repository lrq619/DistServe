"""
Simulate DistServe

Output a JSON (list) where each item is the lifecycle for a request.
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd
import simpy

from simdistserve.base.organize_data import organize_request_df, organize_request_event_df, \
    calculate_per_request_latency, organize_worker_event_df
from simdistserve.base.scheduler import put_requests_with_interarrivals
from simdistserve.base.worker import WorkerConfig
from simdistserve.base.workload import (
    get_gamma_interarrival,
    get_fixed_interarrival,
    convert_absolutearrival_to_interarrival, convert_pd_pair_to_request, sample_requests
)
from simdistserve.clusters.disagg import DisaggCluster
from simdistserve.clusters.vllm import VLLMCluster
from simdistserve.constants import ModelTypes
from simdistserve.estimators.memory_estimator import get_max_num_tokens, is_model_runnable
from simdistserve.base.organize_data import Request_t
from simdistserve.base.request import Request


def parse_args(args_=None):
    parser = argparse.ArgumentParser(description='Simulation: vLLM, DistServe')
    parser.add_argument('--model', type=str, default='facebook/opt-13b',
                        help='Model type (opt_13b, opt_66b, opt_175b,'
                             'or facebook/opt-13b, facebook/opt-66b, facebook/opt-175b)')
    parser.add_argument(
        '--workload-json-str', type=str, default='sharegpt',
        help=(
            "A json string of workload, containing [{'arrival': 0.0, 'prefill_len': 1, 'decode_len': 2}, ...]. "
    ))
    parser.add_argument('--prefill-target', type=int, default=200,
                        help='Target latency for prefill')
    parser.add_argument('--decode-target', type=int, default=100,
                        help='Target latency for decode')

    args = parser.parse_args(args=args_)

    return args


def check_dataset_existence(x):
    if not Path(x).exists():
        raise FileNotFoundError(f"Dataset {x} does not exist.")
    return


def load_workload(workload_json_str: str):
    workload_json = json.loads(workload_json_str)
    requests = []
    arrivals = []
    for i, request_json in enumerate(workload_json):
        r = {"req_id": i,"prefill_length": request_json['prefill_len'], "output_lens": request_json['decode_len']}
        req = Request(**r)
        arrival = request_json['arrival']
        requests.append(req)
        arrivals.append(arrival)

    return requests, arrivals


def main(args, outputs=None):
    outputs = outputs if outputs is not None else {}

    backend = "distserve"
    model_type = ModelTypes.model_str_to_object(args.model)

    TP_Prefill = 1
    PP_prefill = 1
    TP_Decode = 1
    PP_decode = 1

    #
    # Handle vllm in data processing
    #
    if not is_model_runnable(model_type, TP_Prefill, PP_prefill):
        raise ValueError(
            f"Model {model_type} is not runnable with TP={TP_Prefill}, PP={PP_prefill}. "
            f"Skipping by throwing exception..."
        )

    prefill_max_tokens = get_max_num_tokens(model_type, TP_Prefill, PP_prefill)
    decode_max_tokens = get_max_num_tokens(model_type, TP_Decode, PP_decode)

    # Setting the seed to sample request / process
    requests, arrival = load_workload(workload_json_str=args.workload_json_str)

    # Run simulation
    env = simpy.Environment()
    worker_config = WorkerConfig(
        model_type=model_type,
        TP=TP_Prefill, TP_Prefill=TP_Prefill, TP_Decode=TP_Decode,
        prefill_max_batch_size=10 ** 7,  # inf
        decode_max_batch_size=10 ** 7,  # inf
        prefill_max_tokens=prefill_max_tokens,
        decode_max_tokens=decode_max_tokens,
        enable_chunked_prefill=False,
        engine_type=backend,
    )

    cluster = DisaggCluster(
        env=env, PP_prefill=PP_prefill, PP_decode=PP_decode,
        worker_configs=worker_config,
    )
    cluster.run()
    put_requests_with_interarrivals(env, cluster.scheduler, arrival, requests)
    env.run()

    #
    # Collect request-level data and containment
    #
    request_df = organize_request_df(requests)
    request_event_df = organize_request_event_df(requests)
    per_request_latency_df = calculate_per_request_latency(
        request_event_df, request_df.output_lens
    )
    outputs['request_df'] = request_df
    outputs['request_event_df'] = request_event_df
    outputs['per_request_latency_df'] = per_request_latency_df

    workload_duration_ms = max(arrival) - min(arrival)
    print(f"Workload duration: {workload_duration_ms} ms")
    columns = [
        "backend", "model_type", "pd", "rate", "target", "attainment",
        "tp_prefill", "pp_prefill", "tp_decode", "pp_decode",
    ]
    output_results = []
    # Fix the prefill & decode target (SLO & scale),
    # then find the attainment (percentage of requests that meet the SLO)
    # args.slo_scales = 1
    # for scale in args.slo_scales:
    #     prefill_target = args.prefill_target * scale
    #     prefill_attainment = (per_request_latency_df['first_token_latency'] <= prefill_target).sum() / N
    #     prefill_attainment *= 100
    #     item = [args.backend, model_type, 'prefill', rate, prefill_target, prefill_attainment,
    #             TP_Prefill, PP_prefill, TP_Decode, PP_decode]
    #     output_results.append(item)

    #     decode_target = args.decode_target * scale
    #     decode_attainment = (per_request_latency_df['tpot'] <= decode_target).sum() / N
    #     decode_attainment *= 100
    #     item = [args.backend, model_type, 'decode', rate, decode_target, decode_attainment,
    #             TP_Prefill, PP_prefill, TP_Decode, PP_decode]
    #     output_results.append(item)

    #     both_attainment = (
    #                           (per_request_latency_df['first_token_latency'] <= prefill_target) &
    #                           (per_request_latency_df['tpot'] <= decode_target)
    #                       ).sum() / N
    #     both_attainment *= 100
    #     item = [args.backend, model_type, 'both', rate, (prefill_target, decode_target), both_attainment,
    #             TP_Prefill, PP_prefill, TP_Decode, PP_decode]
    #     output_results.append(item)
    #     pass

    # # Fix the attainment (percentage of requests that meet the SLO),
    # # then find the prefill /  decode SLO target that it can meet.
    # slas = args.slas
    # for sla in slas:
    #     prefill_attainment = decode_attainment = sla
    #     prefill_target = per_request_latency_df['first_token_latency'].quantile(prefill_attainment / 100)
    #     decode_target = per_request_latency_df['tpot'].quantile(decode_attainment / 100)
    #     item = [args.backend, model_type, 'prefill', rate, prefill_target, prefill_attainment,
    #             TP_Prefill, PP_prefill, TP_Decode, PP_decode]
    #     output_results.append(item)
    #     item = [args.backend, model_type, 'decode', rate, decode_target, decode_attainment,
    #             TP_Prefill, PP_prefill, TP_Decode, PP_decode]
    #     output_results.append(item)
    #     pass

    # Compute boolean masks
    satisfies_prefill = per_request_latency_df["first_token_latency"] < args.prefill_target
    satisfies_decode = per_request_latency_df["tpot"] < args.decode_target

    # Count how many requests satisfy each SLO
    num_prefill_pass = satisfies_prefill.sum()
    num_decode_pass = satisfies_decode.sum()

    # Convert workload duration to seconds
    workload_duration_s = workload_duration_ms / 1000

    # Compute goodput in req/s
    prefill_goodput_rps = num_prefill_pass / workload_duration_s
    decode_goodput_rps = num_decode_pass / workload_duration_s

    # Print results
    print(f"--- SLO Goodput Report ---")
    print(f"Workload rps: {len(arrival)/workload_duration_s:.1f} req/s")
    print(f"Prefill Goodput: {prefill_goodput_rps:.1f} req/s, {num_prefill_pass} reqs")
    print(f"Decode  Goodput: {decode_goodput_rps:.1f} req/s, {num_decode_pass} reqs")
    print(f"[Simulator Results] Prefill Goodput: {prefill_goodput_rps:.1f} req/s; Decode Goodput: {decode_goodput_rps:.1f} req/s")

    df = pd.DataFrame(output_results, columns=columns)
    outputs['latency_df'] = df

    return


run_experiment = main


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
    # test_opt_13b_grid_search_serial()
    pass

import argparse
import json
import pandas as pd
from typing import List
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

from simdistserve.constants import ModelTypes
from simdistserve.estimators.memory_estimator import get_max_num_tokens, is_model_runnable
from simdistserve.base.request import Request as SimRequest
from simdistserve.base.organize_data import (
    organize_request_df,
    organize_request_event_df,
    calculate_per_request_latency,
)
from simdistserve.base.scheduler import put_requests_with_interarrivals
from simdistserve.base.worker import WorkerConfig
from simdistserve.clusters.disagg import DisaggCluster
import simpy

# -------------------------------
# Argument Parsing for host/port
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, default=8000)
parser.add_argument('--prefill-target-ms', type=int, default=2000)
parser.add_argument('--decode-target-ms', type=int, default=100)
args, _ = parser.parse_known_args()

# -------------------------------
# Workload Structs
# -------------------------------
class WorkloadItem(BaseModel):
    arrival: float
    prefill_len: int
    decode_len: int


class SimulationRequest(BaseModel):
    model: str
    workloads: List[WorkloadItem]

# -------------------------------
# HTTP App Init
# -------------------------------
app = FastAPI()

# -------------------------------
# Helper for Simulation
# -------------------------------
def run_simulation(req: SimulationRequest):
    model_name = req.model
    workloads = req.workloads
    model_type = ModelTypes.model_str_to_object(model_name)
    TP_Prefill = 1
    PP_prefill = 1
    TP_Decode = 1
    PP_decode = 1

    if not is_model_runnable(model_type, TP_Prefill, PP_prefill):
        raise ValueError(f"Model {model_type} not runnable with TP={TP_Prefill}, PP={PP_prefill}")

    prefill_max_tokens = get_max_num_tokens(model_type, TP_Prefill, PP_prefill)
    decode_max_tokens = get_max_num_tokens(model_type, TP_Decode, PP_decode)

    requests = []
    arrivals = []
    for i, item in enumerate(workloads):
        r = {"req_id": i, "prefill_length": item.prefill_len, "output_lens": item.decode_len}
        req = SimRequest(**r)
        requests.append(req)
        arrivals.append(item.arrival)

    env = simpy.Environment()
    worker_config = WorkerConfig(
        model_type=model_type,
        TP=TP_Prefill, TP_Prefill=TP_Prefill, TP_Decode=TP_Decode,
        prefill_max_batch_size=10**7,
        decode_max_batch_size=10**7,
        prefill_max_tokens=prefill_max_tokens,
        decode_max_tokens=decode_max_tokens,
        enable_chunked_prefill=False,
        engine_type="distserve",
    )

    cluster = DisaggCluster(env=env, PP_prefill=PP_prefill, PP_decode=PP_decode, worker_configs=worker_config)
    cluster.run()
    put_requests_with_interarrivals(env, cluster.scheduler, arrivals, requests)
    env.run()

    request_df = organize_request_df(requests)
    request_event_df = organize_request_event_df(requests)
    latency_df = calculate_per_request_latency(request_event_df, request_df.output_lens)
    per_request_latency_df = calculate_per_request_latency(
        request_event_df, request_df.output_lens
    )

    workload_duration_ms = max(arrivals) - min(arrivals)
    print(f"Workload duration: {workload_duration_ms} ms")

    # Compute boolean masks
    satisfies_prefill = per_request_latency_df["first_token_latency"] < args.prefill_target_ms
    satisfies_decode = per_request_latency_df["tpot"] < args.decode_target_ms

    # Count how many requests satisfy each SLO
    num_prefill_pass = satisfies_prefill.sum()
    num_decode_pass = satisfies_decode.sum()

    # Convert workload duration to seconds
    workload_duration_s = workload_duration_ms / 1000

    # Compute goodput in req/s
    prefill_goodput_rps = num_prefill_pass / workload_duration_s
    decode_goodput_rps = num_decode_pass / workload_duration_s

    return {"prefill_goodput": prefill_goodput_rps, "decode_goodput": decode_goodput_rps}

# -------------------------------
# POST /simulate
# -------------------------------
@app.post("/simulate")
async def simulate(simreq: SimulationRequest):
    try:
        result = run_simulation(simreq)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -------------------------------
# Server Entry Point
# -------------------------------
if __name__ == '__main__':
    uvicorn.run("main:app", host=args.host, port=args.port, reload=True)
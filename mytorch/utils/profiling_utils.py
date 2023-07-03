import subprocess
import time
from collections import defaultdict
from typing import Any

import drjit as dr
import psutil
import torch


class TimeProfiler:
    def __init__(self) -> None:
        self.stack = []
        self.count = defaultdict(lambda: 0)
        self.total_time = defaultdict(lambda: 0)
        self.children = defaultdict(set)
        self.enabled = False
        self.synchronize_cuda = True

    def start(self, key: str):
        if not self.enabled:
            return

        parent = None if len(self.stack) == 0 else self.stack[-1][0]
        key = "$$".join([s[0] for s in self.stack] + [key])
        self.children[parent].add(key)
        self.stack.append([key, time.time()])

    def end(self, end_key: str):
        if not self.enabled:
            return

        if len(self.stack) == 0:
            raise ValueError("Stack is empty, missing a call to start()")

        if self.synchronize_cuda:
            torch.cuda.synchronize()
            dr.sync_all_devices()

        key, start_time = self.stack.pop()
        if key.split("$$")[-1] != end_key:
            raise ValueError(f"start key is {key}, but end key is {end_key}; check your start() calls")
        self.total_time[key] += time.time() - start_time
        self.count[key] += 1

    def get_results_string(self):
        if len(self.stack) > 0:
            raise ValueError("Not all profiling has ended!")

        columns = [[], [], [], []]  # key, total, count, avg

        keys = [[k, 0] for k in reversed(sorted(self.children[None]))]
        while len(keys) > 0:
            key, depth = keys.pop()
            keys += list([[k, depth+1] for k in reversed(sorted(self.children[key]))])

            total_time = self.total_time[key]
            count = self.count[key]

            columns[0].append("  " * depth + key.split("$$")[-1])
            columns[1].append(f"{total_time:.2f}s")
            columns[2].append(f"{count}")
            columns[3].append(f"{total_time*1000/count:.2f}ms")

        lengths = [max([len(s) for s in col]) for col in columns]
        return "\n".join(
            [
                " ".join(
                    [f"{columns[0][i]:<{lengths[0]}}"] +
                    [f"{columns[j][i]:>{lengths[j]}}" for j in range(1, len(columns))]
                )
                for i in range(len(columns[0]))
            ]
        )


class VRamProfiler:
    def __init__(self) -> None:
        self.snapshots = []
        self.enabled = False

    def take_snapshot(self, name):
        if not self.enabled:
            return

        data = {
            "name": name,
            "time": time.time(),
        }

        self.get_nvsmi_data(data)
        self.get_torch_data(data)
        self.get_mem_data(data)

        self.snapshots.append(data)

    @staticmethod
    def get_nvsmi_data(data: dict[str, Any]):
        proc = subprocess.run(
            ["nvidia-smi", "--format=csv", "--query-gpu=memory.used"],
            capture_output=True,
            check=True,
        )
        # output (two lines) looks like:
        # memory.used [MiB]
        # 16656 MiB

        # take second line, split
        line = proc.stdout.decode("utf-8").split("\n")[1].split(", ")
        data["nv.used"] = int(line[0].split(" ")[0])

    @staticmethod
    def get_torch_data(data: dict[str, Any]):
        stats = torch.cuda.memory_stats()
        data["torch.allocated"] = stats["allocated_bytes.all.current"]
        data["torch.reserved"] = stats["reserved_bytes.all.current"]
        data["torch.active"] = stats["active_bytes.all.current"]
        data["torch.inactive"] = stats["inactive_split_bytes.all.current"]

    @staticmethod
    def get_mem_data(data: dict[str, Any]):
        mem = psutil.virtual_memory()
        data["mem.total"] = mem.total
        data["mem.available"] = mem.available


class CounterProfiler:
    def __init__(self) -> None:
        self.data: dict[str, list[list[Any]]] = defaultdict(list)
        self.group_counter = 1
        self.enabled = False

    def record(self, name: str, value: Any):
        if not self.enabled:
            return

        gp_list = self.data[name]
        while len(gp_list) < self.group_counter:
            gp_list.append([])
        gp_list[-1].append(value)

    def new_group(self):
        if not self.enabled:
            return

        self.group_counter += 1


time_profiler = TimeProfiler()
vram_profiler = VRamProfiler()
counter_profiler = CounterProfiler()

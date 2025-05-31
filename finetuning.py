# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
from llama_recipes.finetuning import main
import debugpy
import os

if __name__ == "__main__":


    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # debugpy.listen(("0.0.0.0", 5678 + rank))
    # print(f"Waiting for debugger attach on rank {rank}...")
    # debugpy.wait_for_client()
    # print(f"Debugger attached on rank {rank}!")

    fire.Fire(main)
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch


class W8A8CacheConfig:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        default_dir = Path(__file__).resolve().parents[2] / "lmslim" / "configs" / "w8a8"
        self.triton_json_dir = os.getenv("TRITON_JSON_DIR", str(default_dir))
        self.triton_json_dict = {}
        self.triton_moejson_dict = {}
        self.triton_json_list = []
        self.weight_shapes = []
        self.moe_weight_shapes = []

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch_name = getattr(props, "gcnArchName", "unknown").split(":")[0]
        self.device_name = f"{arch_name}_{props.multi_processor_count}cu"
        self.topk = 1
        self.quant_method = None

    def gen_model_json(
        self, num_experts: Optional[int] = 0, block_size: Optional[list] = None
    ) -> None:
        del num_experts
        json_dir = os.getenv("LMSLIM_TUNING_JSON", "None")
        if json_dir == "None" or not os.path.exists(json_dir):
            return

        config = {
            "layers": {
                "linear": {"shapes": list(self.weight_shapes), "m_range": "None"},
                "moe": {
                    "shapes": [
                        {"E": shape[0], "N1": shape[1], "N2": shape[2], "K": shape[3]}
                        for shape in self.moe_weight_shapes
                        if len(shape) == 4
                    ],
                    "m_range": "None",
                    "topk": self.topk,
                },
            },
            "quantization_config": {
                "quant_method": self.quant_method,
                "weight_block_size": block_size if block_size is not None else "None",
            },
        }
        with open(os.path.join(json_dir, "model.json"), "w") as file:
            json.dump(config, file, indent=4)

    @staticmethod
    def getspec_config(configs_dict, M, N, K):
        return configs_dict.get(f"{M}_{N}_{K}")

    @staticmethod
    def get_triton_cache(file_path, n, k):
        del n, k
        return W8A8CacheConfig._read_nested_cache(file_path)

    def get_w8a8json_name(self, n, k):
        return os.path.join(
            self.triton_json_dir, f"W8A8_{n}_{k}_{self.device_name}.json"
        )

    @staticmethod
    def get_blockint8_triton_cache(file_path, n, k, block_n, block_k):
        del n, k, block_n, block_k
        return W8A8CacheConfig._read_nested_cache(file_path)

    def get_blockint8json_name(self, n, k, block_n, block_k):
        return os.path.join(
            self.triton_json_dir,
            f"BLOCKINT8_{n}_{k}_{block_n}_{block_k}_{self.device_name}.json",
        )

    def get_moeint8json_name(
        self,
        E,
        N1,
        N2,
        K,
        TOPK,
        block_size: Optional[list] = None,
        use_int4_w4a8: bool = False,
    ):
        if use_int4_w4a8:
            prefix = "MOE_W4A8INT8"
        elif block_size is not None:
            prefix = "MOE_BLOCKINT8"
        else:
            prefix = "MOE_W8A8INT8"
        block_suffix = (
            f"[{block_size[0]},{block_size[1]}]" if block_size is not None else ""
        )
        return os.path.join(
            self.triton_json_dir,
            f"{prefix}{block_suffix}_E={E}_N1={N1}_N2={N2}_K={K}_TOPK{TOPK}_{self.device_name}.json",
        )

    @staticmethod
    def get_moeint8_triton_cache(file_path, E, N1, N2, K, TOPK):
        del E, N1, N2, K, TOPK
        return W8A8CacheConfig._read_nested_cache(file_path)

    @staticmethod
    def _read_nested_cache(file_path):
        if not os.path.exists(file_path):
            return None
        with open(file_path) as file:
            cache_data = json.load(file)
        return {
            f"{sub_key}_{key}": sub_value
            for key, value in cache_data.items()
            for sub_key, sub_value in value.items()
        }


W8a8GetCacheJSON = W8A8CacheConfig

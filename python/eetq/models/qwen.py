import torch.nn as nn
import torch

from eetq.utils import replace_fused_gateup, replace_split_gateup, split_tp_column, split_tp_row, merge_tp_handler
from eetq.modules import W8A16Linear
from .base import BaseEETQForCausalLM

class QwenEETQForCausalLM(BaseEETQForCausalLM):

    def fuse_layers(self, tp):
        self.tp = tp
        self.fuser = QwenFuser(self.model)
        print("[EET][INFO] fusing qkv and gateup ...")
        self.fuser.fuse_gateup()
        if self.tp > 1:
            print("[EET][INFO] spliting tp ...")
            self.fuser.split_tp(self.tp)

    def split_layers(self):
        if self.tp > 1:
            print("[EET][INFO] merging tp ...")
            self.fuser.merge_tp()
        print("[EET][INFO] spliting qkv and gateup ...")
        self.fuser.split_gateup()

class QwenFuser:
    def __init__(self, model):
        self.model = model


    def fuse_gateup(self):
        device = self.model.device

        all_gateup = [[None, None] for i in range(self.model.config.num_hidden_layers)]

        gateup_index_map = {"w2": 0, "w1": 1}

        self.gateup_index_map = [[0, 0] for i in range(self.model.config.num_hidden_layers)]
        for name, m in self.model.named_modules():
            if type(m) in [nn.Linear] and name != "lm_head":
                levels = name.split(".")
                num_layers = int(levels[2])
                linear_name = levels[4]
                if linear_name in ["w1", "w2"]:
                    all_gateup[num_layers][gateup_index_map[linear_name]] = m


        for _, gateup in enumerate(all_gateup):
            self.gateup_index_map[_][1] = gateup[0].weight.shape[0]
            name = f"transformer.h.{_}.mlp.gateup_proj"
            gateup_weight = [x.weight for x in gateup]
            gateup_weight = torch.cat(gateup_weight, dim=0)
            fused_gateup = nn.Linear(gateup_weight.shape[1], gateup_weight.shape[0])
            fused_gateup.weight = nn.Parameter(gateup_weight.to(device), requires_grad=False)
            replace_fused_gateup(self.model, name, fused_gateup, gate_proj="w2", up_proj="w1")

    def split_gateup(self):
        modules = [(name, m) for name, m in self.model.named_modules()]
        for name, m in modules:
            if type(m) in [W8A16Linear] and name != "lm_head":
                levels = name.split(".")
                num_layers = int(levels[2])
                linear_name = levels[4]
                if linear_name == "gateup_proj":
                    replace_split_gateup(self.model, name, m, self.gateup_index_map[num_layers], gate_proj="w2", up_proj="w1")

    def split_tp(self, tp=2):
        raise NotImplementedError("QwenEETQForCausalLM does not support split_tp")

    def merge_tp(self):
        raise NotImplementedError("QwenEETQForCausalLM does not support merge_tp")

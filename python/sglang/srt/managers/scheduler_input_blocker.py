# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from enum import Enum, auto
from typing import List, Optional, Any

import torch
from sglang.srt.managers.io_struct import BlockReqInput, BlockReqType


class SchedulerInputBlocker:
    def __init__(self, noop: bool):
        self._state = _State.UNBLOCKED
        self._pending_reqs = []
        self._noop = noop

    def handle(self, recv_reqs: Optional[List[Any]]):
        assert (recv_reqs is None) == self._noop

        if not self._noop:
            output_reqs = []
            for recv_req in recv_reqs:
                output_reqs += self._handle_recv_req(recv_req)

        self._maybe_fulfill_awaiting_global_unblock()

        if not self._noop:
            return output_reqs

    def _handle_recv_req(self, recv_req):
        if isinstance(recv_req, BlockReqInput):
            if recv_req.type == BlockReqType.BLOCK:
                self._execute_block_req()
                return []
            elif recv_req.type == BlockReqType.UNBLOCK:
                self._execute_unblock_req()
                return []
            else:
                raise NotImplementedError(f"{recv_req=}")
        else:
            if self._state == _State.UNBLOCKED:
                return [recv_req]
            else:
                self._pending_reqs.append(recv_req)
                return []

    def _execute_block_req(self):
        self._change_state(original=_State.UNBLOCKED, target=_State.BLOCKED)

    def _execute_unblock_req(self):
        self._change_state(original=_State.BLOCKED, target=_State.AWAITING_GLOBAL_UNBLOCK)

    def _maybe_fulfill_awaiting_global_unblock(self):
        if self._noop:
            local_fulfill = True
        else:
            local_fulfill = TODO

        global_fulfill = torch.distributed.all_reduce(torch.tensor(local_fulfill),
                                                      torch.distributed.ReduceOp.MIN).item()

        TODO

    def _change_state(self, original: "_State", target: "_State"):
        assert self._state == original, f"{self._state=} {original=} {target=}"
        self._state = target


class _State(Enum):
    UNBLOCKED = auto()
    BLOCKED = auto()
    AWAITING_GLOBAL_UNBLOCK = auto()

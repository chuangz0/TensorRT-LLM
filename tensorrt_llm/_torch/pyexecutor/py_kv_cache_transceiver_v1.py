import concurrent
import pickle
import sys
import threading
import time
import uuid
from abc import ABC
from dataclasses import dataclass
from typing import Optional

import torch
import zmq

import tensorrt_llm.bindings
import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm import Mapping, SamplingParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes
from tensorrt_llm.bindings import DataType
from tensorrt_llm.disaggregated_params import DisaggregatedParams

AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
LlmRequestType = tensorrt_llm.bindings.internal.batch_manager.LlmRequestType
DataType = tensorrt_llm.bindings.DataType

nixl_path = "/opt/nvidia/nvda_nixl/lib/python3/dist-packages"
if nixl_path not in sys.path:
    sys.path.insert(0, nixl_path)

from nixl._api import nixl_agent, nixl_agent_config, nixl_xfer_handle


def get_local_ip():
    """
    Lupin-style local IP detection - smart and reliable approach
    """
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        if not local_ip.startswith("127."):
            return local_ip
    except:
        pass

    try:
        import netifaces
        for interface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                for addr in addrs[netifaces.AF_INET]:
                    ip = addr['addr']
                    if not ip.startswith("127.") and not ip.startswith(
                            "169.254"):
                        return ip
    except ImportError:
        pass

    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        if not local_ip.startswith("127."):
            return local_ip
    except:
        pass
    return "127.0.0.1"


class TransferOp:
    READ = "READ"
    WRITE = "WRITE"


class MemoryDesc:

    def __init__(self, ptr: int, size: int, device_id: int):
        self.ptr = ptr
        self.size = size
        self.device_id = device_id


class MemoryDescs:

    def __init__(self, type, descs: list[tuple[int, int, int] | MemoryDesc]):
        self.type = type
        self.descs = descs


class RegMemoryDescs:

    def __init__(self, type, descs: list[tuple[int, int, int, str]]):
        self.type = type
        self.descs = descs


class TransferRequest:

    def __init__(self, op: TransferOp, src_descs: MemoryDescs,
                 dst_descs: MemoryDescs, remote_name: str, sync_message: str):
        self.op = op
        self.src_descs = src_descs
        self.dst_descs = dst_descs
        self.remote_name = remote_name
        self.sync_message = sync_message


class TransferStatus:

    def __init__(self):
        pass

    def is_completed(self):
        raise NotImplementedError

    def wait(self):
        raise NotImplementedError


class BaseTransferAgent(ABC):

    def register_memory(self, descs: MemoryDescs):
        raise NotImplementedError

    def deregister_memory(self, descs: MemoryDescs):
        raise NotImplementedError

    def load_remote_agent(self, name: str, agent_desc: str):
        raise NotImplementedError

    def get_local_agent_desc(self):
        raise NotImplementedError

    def invalidate_remote_agent(self, name: str):
        raise NotImplementedError

    def submit_transfer_requests(self,
                                 request: TransferRequest) -> TransferStatus:
        raise NotImplementedError

    def notify_sync_message(self, name: str, sync_message: str):
        raise NotImplementedError

    def check_remote_descs(self, name: str, memory_descs: list[int]):
        raise NotImplementedError


class NixlTransferStatus(TransferStatus):

    def __init__(self, agent: nixl_agent, handle: nixl_xfer_handle):
        self.agent = agent
        self.handle = handle

    def is_completed(self):
        status = self.agent.check_xfer_state(self.handle)
        return status == "DONE"

    def wait(self):
        status = "PROC"
        while status == "PROC":
            status = self.agent.check_xfer_state(self.handle)
            if status == "ERR":
                return False  # transfer failed
            # sleep(0.1)
        return True


class NixlTransferAgent(BaseTransferAgent):

    def __init__(self, name: str, use_prog_thread: bool):
        self.name = name
        agent_config = nixl_agent_config(enable_prog_thread=use_prog_thread,
                                         backends=["UCX"])
        self.agent = nixl_agent(name, agent_config)

    def register_memory(self, descs: RegMemoryDescs):
        reg_descs = self.agent.get_reg_descs(descs.descs, descs.type)
        self.agent.register_memory(reg_descs)

    def deregister_memory(self, descs: RegMemoryDescs):
        self.agent.deregister_memory(descs.descs, descs.type)

    def load_remote_agent(self, name: str, agent_desc: bytes):
        self.agent.add_remote_agent(agent_desc)

    def get_local_agent_desc(self):
        return self.agent.get_agent_metadata()

    def invalidate_remote_agent(self, name: str):
        self.agent.remove_remote_agent(name)

    def submit_transfer_requests(self,
                                 request: TransferRequest) -> TransferStatus:
        src_xfer_descs = self.agent.get_xfer_descs(request.src_descs.descs,
                                                   request.src_descs.type)
        dst_xfer_descs = self.agent.get_xfer_descs(request.dst_descs.descs,
                                                   request.dst_descs.type)
        handle = self.agent.initialize_xfer(request.op, src_xfer_descs,
                                            dst_xfer_descs, request.remote_name,
                                            request.sync_message)
        status = self.agent.transfer(handle)
        assert status != "ERR"
        return NixlTransferStatus(self.agent, handle)


@dataclass
class InstanceRankInfo:
    instance_name: str
    instance_rank: int
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    dp_size: int
    dp_rank: int
    cp_size: int
    cp_rank: int
    kv_head_num_per_rank: int
    #  [numLayers,kv_factor,heads,tokens,dimsPerHead]
    tokens_per_block: int
    dims_per_head: int
    element_size: int
    enable_attention_dp: bool
    is_mla: bool
    layer_num_per_pp: list[int]
    kvcache_ptrs: list[int]
    aux_ptrs: list[int]
    server_endpoint: str
    recv_endpoint: str
    transfer_engine_info: bytes


@dataclass
class InstanceInfo:
    instance_name: str
    tp_size: int
    pp_size: int
    dp_size: int
    cp_size: int
    kv_head_num_per_rank: int
    tokens_per_block: int
    dims_per_head: int
    element_size: int
    enable_attention_dp: bool
    is_mla: bool
    layer_num_per_pp: list[int]
    ctx_server_endpoints: list[str]


@dataclass
class peerDomainTargets:
    domain_pp_size: int
    domain_tp_size: int
    domain_cp_size: int
    duplicate_head_factor: int
    peer_duplicate_head_factor: int
    target_peer_pp_layer_num: list[int]
    ranks: list[int]


class ResourceRegistrar:

    def __init__(self, instance_rank_info: InstanceRankInfo,
                 instance_info: InstanceInfo):
        self.instance_rank_info = instance_rank_info
        self.instance_info = instance_info
        self.peer_instance_rank_info_cache = {}
        self.peer_domain_targets_cache = {}

        self.kv_block_ptrs_extractor_cache = {}

    def register_peer_info(self, peer_instance_name: str, peer_rank: int,
                           peer_instance_rank_info: InstanceRankInfo):
        self.peer_instance_rank_info_cache[
            peer_instance_name + str(peer_rank)] = peer_instance_rank_info

    def unregister_peer_info(self, peer_instance_name: str, peer_rank: int):
        del self.peer_instance_rank_info_cache[peer_instance_name +
                                               str(peer_rank)]

    def get_peer_instance_rank_info(self, peer_instance_name: str,
                                    peer_rank: int):
        return self.peer_instance_rank_info_cache[peer_instance_name +
                                                  str(peer_rank)]

    def get_instance_info(self):
        return self.instance_info

    def get_instance_rank_info(self):
        return self.instance_rank_info

    def get_peer_domain_targets(self, peer_instance_info: InstanceInfo,
                                peer_dp_rank: int):

        # return the mapping rank relationship between self and peer in asymmetric parallelism case

        if peer_instance_info.instance_name + str(
                peer_dp_rank) in self.peer_domain_targets_cache:
            return self.peer_domain_targets_cache[
                peer_instance_info.instance_name + str(peer_dp_rank)]
        peer_pp_rank_start = 0
        peer_pp_rank_end = 0
        self_start_layer_id = sum(
            self.instance_rank_info.layer_num_per_pp[:self.instance_rank_info.
                                                     pp_rank])
        self_end_layer_id = self_start_layer_id + self.instance_rank_info.layer_num_per_pp[
            self.instance_rank_info.pp_rank]
        pre_peer_pp_layer_id = 0
        target_peer_pp_ranks = []
        target_peer_pp_layer_num = []
        for pp_rank in range(peer_instance_info.pp_size):
            peer_pp_start_layer_id = pre_peer_pp_layer_id
            peer_pp_end_layer_id = peer_pp_start_layer_id + peer_instance_info.layer_num_per_pp[
                pp_rank]
            if self_start_layer_id < peer_pp_end_layer_id and self_end_layer_id > peer_pp_start_layer_id:
                target_peer_pp_ranks.append(pp_rank)
                target_peer_pp_layer_num.append(
                    min(peer_pp_end_layer_id, self_end_layer_id) -
                    max(peer_pp_start_layer_id, self_start_layer_id))
            pre_peer_pp_layer_id += peer_instance_info.layer_num_per_pp[pp_rank]

        peer_pp_rank_start = target_peer_pp_ranks[0]
        domain_pp_size = len(target_peer_pp_ranks)
        peer_pp_rank_end = peer_pp_rank_start + len(target_peer_pp_ranks)

        # dp
        self_tp_size_per_dp_group = self.instance_rank_info.tp_size / self.instance_rank_info.dp_size if self.instance_rank_info.enable_attention_dp else self.instance_rank_info.tp_size
        peer_tp_size_per_dp_group = peer_instance_info.tp_size / peer_instance_info.dp_size if peer_instance_info.enable_attention_dp else peer_instance_info.tp_size
        self_tprank_in_dp_group = self.instance_rank_info.tp_rank % self_tp_size_per_dp_group
        peer_tp_rank_start = 0
        peer_tp_rank_end = 0
        domain_tp_size = 1
        if (self_tp_size_per_dp_group <= peer_tp_size_per_dp_group):
            domain_tp_size = peer_tp_size_per_dp_group // self_tp_size_per_dp_group
            peer_tp_rank_start = self_tprank_in_dp_group * domain_tp_size + peer_dp_rank * peer_tp_size_per_dp_group
            peer_tp_rank_end = peer_tp_rank_start + domain_tp_size
        else:
            peer_tp_rank_start = self_tprank_in_dp_group // (
                self_tp_size_per_dp_group // peer_tp_size_per_dp_group
            ) + peer_dp_rank * peer_tp_size_per_dp_group
            peer_tp_rank_end = peer_tp_rank_start + domain_tp_size
        # peer_tp_rank_start = self_tprank_in_dp_group * domain_tp_size + peer_dp_rank * peer_tp_size_per_dp_group
        # peer_tp_rank_end = peer_tp_rank_start + domain_tp_size
        peer_cp_rank_start = 0
        peer_cp_rank_end = 0
        domain_cp_size = 1
        if (self.instance_rank_info.cp_size <= peer_instance_info.cp_size):
            domain_cp_size = peer_instance_info.cp_size // self.instance_rank_info.cp_size
            peer_cp_rank_start = self.instance_rank_info.cp_rank * domain_cp_size
            peer_cp_rank_end = peer_cp_rank_start + domain_cp_size
        else:
            peer_cp_rank_start = self.instance_rank_info.cp_rank // (
                self.instance_rank_info.cp_size // peer_instance_info.cp_size)
            peer_cp_rank_end = peer_cp_rank_start + domain_cp_size
        ranks = []
        for pp_rank in range(peer_pp_rank_start, peer_pp_rank_end):

            for cp_rank in range(peer_cp_rank_start, peer_cp_rank_end):
                for tp_rank in range(peer_tp_rank_start, peer_tp_rank_end):
                    ranks.append(pp_rank * peer_instance_info.tp_size *
                                 peer_instance_info.cp_size +
                                 cp_rank * peer_instance_info.tp_size + tp_rank)
        duplicate_head_factor = max(
            1, self.instance_rank_info.kv_head_num_per_rank *
            self_tp_size_per_dp_group //
            (peer_instance_info.kv_head_num_per_rank *
             peer_tp_size_per_dp_group))
        peer_duplicate_head_factor = max(
            1, peer_instance_info.kv_head_num_per_rank *
            peer_tp_size_per_dp_group //
            (self.instance_rank_info.kv_head_num_per_rank *
             self_tp_size_per_dp_group))

        peer_domain_ranks = peerDomainTargets(
            domain_pp_size=domain_pp_size,
            domain_tp_size=domain_tp_size,
            domain_cp_size=domain_cp_size,
            duplicate_head_factor=duplicate_head_factor,
            peer_duplicate_head_factor=peer_duplicate_head_factor,
            target_peer_pp_layer_num=target_peer_pp_layer_num,
            ranks=ranks,
        )
        self.peer_domain_targets_cache[peer_instance_info.instance_name +
                                       str(peer_dp_rank)] = peer_domain_ranks
        return peer_domain_ranks

    def get_kv_block_ptrs_extractor(self,
                                    peer_instance_rank_info: InstanceRankInfo):

        if peer_instance_rank_info.instance_name + str(
                peer_instance_rank_info.instance_rank
        ) in self.kv_block_ptrs_extractor_cache:
            return self.kv_block_ptrs_extractor_cache[
                peer_instance_rank_info.instance_name +
                str(peer_instance_rank_info.instance_rank)]

        # return the kv block ptrs extractor for the peer. the extractor will be used to extract the kv block ptrs and submit to transfer agent.
        # the returned will be a callable object that will be called to extract the kv block ptrs.
        def extractor(
            src_kv_block_ptrs: list[int],
            src_kv_block_size: int,
            dst_kv_block_ptrs: list[int],
            dst_kv_block_size: int,
        ) -> tuple[list[int], int, list[int], int]:
            # return the kv block ptrs and size for the peer.
            return src_kv_block_ptrs, src_kv_block_size, dst_kv_block_ptrs, dst_kv_block_size

        # TODO: implement the extractor

        self_layer_num = self.instance_rank_info.layer_num_per_pp[
            self.instance_rank_info.pp_rank]
        peer_layer_num = peer_instance_rank_info.layer_num_per_pp[
            peer_instance_rank_info.pp_rank]
        kv_factor = 1 if self.instance_rank_info.is_mla else 2
        # just ruturn the src_kv_block_ptrs and dst_kv_block_ptrs if tp and pp are the same
        self_tp_size_per_dp_group = self.instance_rank_info.tp_size // self.instance_rank_info.dp_size if self.instance_rank_info.enable_attention_dp else self.instance_rank_info.tp_size
        peer_tp_size_per_dp_group = peer_instance_rank_info.tp_size // peer_instance_rank_info.dp_size if peer_instance_rank_info.enable_attention_dp else peer_instance_rank_info.tp_size
        self_tprank_in_dp_group = self.instance_rank_info.tp_rank % self_tp_size_per_dp_group
        peer_tprank_in_dp_group = peer_instance_rank_info.tp_rank % peer_tp_size_per_dp_group
        is_duplicate_head = self.instance_rank_info.kv_head_num_per_rank * self_tp_size_per_dp_group != peer_instance_rank_info.kv_head_num_per_rank * peer_tp_size_per_dp_group
        write_all_head = is_duplicate_head or self.instance_rank_info.is_mla or self_tp_size_per_dp_group == peer_tp_size_per_dp_group
        if write_all_head and self.instance_rank_info.pp_size == peer_instance_rank_info.pp_size:

            def extractor(
                src_kv_block_ptrs: list[int],
                src_kv_block_size: int,
                dst_kv_block_ptrs: list[int],
                dst_kv_block_size: int,
            ) -> tuple[list[int], int, list[int], int]:
                return src_kv_block_ptrs, src_kv_block_size, dst_kv_block_ptrs, dst_kv_block_size

            self.kv_block_ptrs_extractor_cache[
                peer_instance_rank_info.instance_name +
                str(peer_instance_rank_info.instance_rank)] = extractor
            print(f" call get_kv_block_ptrs_extractor end with extractor 0")
            return extractor

        # assume
        # selfStartLayer, selfEndLayer,  , peerStartLayer, peerEndLayer
        src_start_layer_id = sum(
            self.instance_rank_info.layer_num_per_pp[:self.instance_rank_info.
                                                     pp_rank])
        src_end_layer_id = src_start_layer_id + self.instance_rank_info.layer_num_per_pp[
            self.instance_rank_info.pp_rank]
        peer_start_layer_id = sum(
            peer_instance_rank_info.layer_num_per_pp[:peer_instance_rank_info.
                                                     pp_rank])
        peer_end_layer_id = peer_start_layer_id + peer_instance_rank_info.layer_num_per_pp[
            peer_instance_rank_info.pp_rank]
        start_layer_id = max(src_start_layer_id, peer_start_layer_id)
        end_layer_id = min(src_end_layer_id, peer_end_layer_id)
        transfer_layer_num = end_layer_id - start_layer_id
        src_start_layer_offset = start_layer_id - src_start_layer_id
        peer_start_layer_offset = start_layer_id - peer_start_layer_id
        if write_all_head:  # write all heads but pp size are different
            fragment_size = transfer_layer_num * kv_factor * self.instance_rank_info.kv_head_num_per_rank * self.instance_rank_info.tokens_per_block * self.instance_rank_info.dims_per_head * self.instance_rank_info.element_size
            src_block_offset = src_start_layer_offset * kv_factor * self.instance_rank_info.kv_head_num_per_rank * self.instance_rank_info.tokens_per_block * self.instance_rank_info.dims_per_head * self.instance_rank_info.element_size
            dst_block_offset = peer_start_layer_offset * kv_factor * peer_instance_rank_info.kv_head_num_per_rank * peer_instance_rank_info.tokens_per_block * peer_instance_rank_info.dims_per_head * peer_instance_rank_info.element_size

            def extractor(
                src_kv_block_ptrs: list[int],
                src_kv_block_size: int,
                dst_kv_block_ptrs: list[int],
                dst_kv_block_size: int,
                s_fragment_size: int = fragment_size,
                ssrc_block_offset: int = src_block_offset,
                ddst_block_offset: int = dst_block_offset,
            ) -> tuple[list[int], int, list[int], int]:
                src_kv_blocks_transfer_ptrs = [
                    src_kv_block_ptr + ssrc_block_offset
                    for src_kv_block_ptr in src_kv_block_ptrs
                ]
                dst_kv_blocks_transfer_ptrs = [
                    dst_kv_block_ptr + ddst_block_offset
                    for dst_kv_block_ptr in dst_kv_block_ptrs
                ]
                return src_kv_blocks_transfer_ptrs, s_fragment_size, dst_kv_blocks_transfer_ptrs, s_fragment_size

            self.kv_block_ptrs_extractor_cache[
                peer_instance_rank_info.instance_name +
                str(peer_instance_rank_info.instance_rank)] = extractor
            return extractor
            # head num are different
        head_fragment_size = self.instance_rank_info.tokens_per_block * self.instance_rank_info.dims_per_head * self.instance_rank_info.element_size
        continue_heads_fragment_size = min(
            self.instance_rank_info.kv_head_num_per_rank,
            peer_instance_rank_info.kv_head_num_per_rank) * head_fragment_size
        self.instance_rank_info.kv_head_num_per_rank * head_fragment_size
        peer_instance_rank_info.kv_head_num_per_rank * head_fragment_size
        src_head_offset = 0
        dst_head_offset = 0
        if (self_tp_size_per_dp_group < peer_tp_size_per_dp_group):
            src_head_offset = peer_tprank_in_dp_group % (
                peer_tp_size_per_dp_group //
                self_tp_size_per_dp_group) * continue_heads_fragment_size
            dst_head_offset = 0
        if (self_tp_size_per_dp_group > peer_tp_size_per_dp_group):
            dst_head_offset = self_tprank_in_dp_group % (
                self_tp_size_per_dp_group //
                peer_tp_size_per_dp_group) * continue_heads_fragment_size
            src_head_offset = 0

        src_layer_kv_ele_size = self.instance_rank_info.kv_head_num_per_rank * self.instance_rank_info.tokens_per_block * self.instance_rank_info.dims_per_head * self.instance_rank_info.element_size
        dst_layer_kv_ele_size = peer_instance_rank_info.kv_head_num_per_rank * peer_instance_rank_info.tokens_per_block * peer_instance_rank_info.dims_per_head * peer_instance_rank_info.element_size
        src_layer_ele_size = src_layer_kv_ele_size * kv_factor
        dst_layer_ele_size = dst_layer_kv_ele_size * kv_factor
        print(
            f" src_layer_ele_size: {src_layer_ele_size}, dst_layer_ele_size: {dst_layer_ele_size}"
        )

        def extractor(
            src_kv_block_ptrs: list[int],
            src_kv_block_size: int,
            dst_kv_block_ptrs: list[int],
            dst_kv_block_size: int,
            tt_transfer_layer_num: int = transfer_layer_num,
            ssrc_head_offset: int = src_head_offset,
            ssrc_layer_ele_size: int = src_layer_ele_size,
            ssrc_layer_kv_ele_size: int = src_layer_kv_ele_size,
            ddst_head_offset: int = dst_head_offset,
            sstart_layer_id: int = start_layer_id,
            ddst_layer_ele_size: int = dst_layer_ele_size,
            ddst_layer_kv_ele_size: int = dst_layer_kv_ele_size,
            ssrc_start_layer_offset: int = src_start_layer_offset,
            ddst_start_layer_offset: int = peer_start_layer_offset,
            s_continue_heads_fragment_size: int = continue_heads_fragment_size,
        ) -> tuple[list[int], int, list[int], int]:

            block_num = len(src_kv_block_ptrs)
            src_kv_blocks_transfer_ptrs = []
            dst_kv_blocks_transfer_ptrs = []
            for block_id in range(block_num):
                src_kv_block_ptr = src_kv_block_ptrs[block_id]
                dst_kv_block_ptr = dst_kv_block_ptrs[block_id]
                for layer_id in range(tt_transfer_layer_num):
                    src_layer_id = sstart_layer_id + layer_id - ssrc_start_layer_offset
                    dst_layer_id = sstart_layer_id + layer_id - ddst_start_layer_offset
                    for kv in range(kv_factor):
                        src_kv_block_transfer_ptr = src_kv_block_ptr + ssrc_layer_ele_size * src_layer_id + ssrc_layer_kv_ele_size * kv + ssrc_head_offset
                        dst_kv_block_transfer_ptr = dst_kv_block_ptr + ddst_layer_ele_size * dst_layer_id + ddst_layer_kv_ele_size * kv + ddst_head_offset
                        src_kv_blocks_transfer_ptrs.append(
                            src_kv_block_transfer_ptr)
                        dst_kv_blocks_transfer_ptrs.append(
                            dst_kv_block_transfer_ptr)
            return src_kv_blocks_transfer_ptrs, s_continue_heads_fragment_size, dst_kv_blocks_transfer_ptrs, s_continue_heads_fragment_size

        self.kv_block_ptrs_extractor_cache[
            peer_instance_rank_info.instance_name +
            str(peer_instance_rank_info.instance_rank)] = extractor
        return extractor


class TransferSession:

    def __init__(self, kv_cache_manager: KVCacheManager, request_id: int,
                 resource_register: ResourceRegistrar):
        self.kv_cache_manager = kv_cache_manager
        self.request_id = request_id
        self.resource_register = resource_register
        self.future = concurrent.futures.Future()
        self.first_extracted = False
        self.encountered_count = 0
        self.expect_count = 0
        self.session_id = str(uuid.uuid4())

    def get_future_for_session(self):
        return self.future

    def extra_trans_meta(self):
        raise NotImplementedError("Not implemented")


@dataclass
class TransferGenSideReqInfo:
    ctx_req_id: int
    instance_name: str
    instance_rank: int
    block_ids: list[int]
    start_token_id: int
    gen_req_id: int
    session_id: str


@dataclass
class AgentSendArgs:
    session_id: str
    future_for_session: concurrent.futures.Future
    src_kv_ptrs: list[int]
    dst_kv_ptrs: list[int]
    kv_sizes: list[int]
    expect_count: int
    remote_name: str
    src_aux_ptrs: list[int] = None
    dst_aux_ptrs: list[int] = None
    aux_sizes: list[int] = None
    peer_endpoint: Optional[str] = None  # used for send state
    peer_session_id: Optional[str] = None


@dataclass
class AgentRecvArgs:
    session_id: str
    future_for_session: concurrent.futures.Future
    expect_count: int
    remote_name: str


class SessionState:

    INIT = "INIT"
    WAITING_FOR_SEND = "WAITING_FOR_SEND"
    TRANSFERRING = "TRANSFERRING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class MessageType:
    TERMINATION = "TERMINATION"
    TASK_STATE = "TASK_STATE"
    PEER_INFO = "PEER_INFO"
    REQUEST_DATA = "REQUEST_DATA"
    REQUEST_INSTANCE_INFO = "REQUEST_INSTANCE_INFO"
    REGISTER_RANK_INFO = "REQUEST_RANK_INFO"


class SenderSession(TransferSession):

    def __init__(self, kv_cache_manager: KVCacheManager, request_id: int,
                 resource_register: ResourceRegistrar):
        super().__init__(kv_cache_manager, request_id, resource_register)

    def extract_trans_meta(self,
                           dst_info: TransferGenSideReqInfo) -> AgentSendArgs:
        peer_instance_rank_info = self.resource_register.get_peer_instance_rank_info(
            dst_info.instance_name, dst_info.instance_rank)
        peer_domain_targets = self.resource_register.get_peer_domain_targets(
            peer_instance_rank_info, peer_instance_rank_info.dp_rank)
        expect_count = len(peer_domain_targets.ranks)
        if not self.first_extracted:
            self.first_extracted = True
            self.expect_count = expect_count
        self.encountered_count = self.encountered_count + 1
        if not self._need_send_transfer(peer_domain_targets,
                                        peer_instance_rank_info):
            return AgentSendArgs(
                session_id=self.session_id,
                future_for_session=self.future,
                src_kv_ptrs=[],
                dst_kv_ptrs=[],
                kv_sizes=[],
                expect_count=expect_count,
                remote_name=peer_instance_rank_info.instance_name +
                str(peer_instance_rank_info.instance_rank),
                peer_endpoint=peer_instance_rank_info.recv_endpoint,
                peer_session_id=self.session_id)
        kv_factor = 1 if self.resource_register.get_instance_rank_info(
        ).is_mla else 2
        self_kv_block_size = self.resource_register.get_instance_rank_info(
        ).layer_num_per_pp[self.resource_register.get_instance_rank_info(
        ).pp_rank] * kv_factor * self.resource_register.get_instance_rank_info(
        ).kv_head_num_per_rank * self.resource_register.get_instance_rank_info(
        ).tokens_per_block * self.resource_register.get_instance_rank_info(
        ).dims_per_head * self.resource_register.get_instance_rank_info(
        ).element_size
        peer_kv_block_size = peer_instance_rank_info.layer_num_per_pp[
            peer_instance_rank_info.
            pp_rank] * kv_factor * peer_instance_rank_info.kv_head_num_per_rank * peer_instance_rank_info.tokens_per_block * peer_instance_rank_info.dims_per_head * peer_instance_rank_info.element_size

        dst_block_ids = dst_info.block_ids

        src_block_ids = self.kv_cache_manager.get_batch_cache_indices(
            [self.request_id])[0]
        src_block_ids, dst_block_ids = self._filter_kv_blocks(
            src_block_ids, dst_block_ids)
        src_kv_ptr = self.resource_register.get_instance_rank_info(
        ).kvcache_ptrs[0]
        dst_kv_ptr = peer_instance_rank_info.kvcache_ptrs[0]
        src_kv_blocks_ptrs = [
            src_kv_ptr + self_kv_block_size * block_id
            for block_id in src_block_ids
        ]
        dst_kv_blocks_ptrs = [
            dst_kv_ptr + peer_kv_block_size * block_id
            for block_id in dst_block_ids
        ]
        print(
            f" call extract_trans_meta src_kv_blocks_ptrs: {src_kv_blocks_ptrs} self_kv_block_size: {self_kv_block_size} dst_kv_blocks_ptrs: {dst_kv_blocks_ptrs} peer_kv_block_size: {peer_kv_block_size}"
        )
        extractor = self.resource_register.get_kv_block_ptrs_extractor(
            peer_instance_rank_info)
        src_kv_blocks_transfer_ptrs, src_kv_blocks_size, dst_kv_blocks_transfer_ptrs, dst_kv_blocks_size = extractor(
            src_kv_blocks_ptrs, self_kv_block_size, dst_kv_blocks_ptrs,
            peer_kv_block_size)

        return AgentSendArgs(
            session_id=self.session_id,
            future_for_session=self.future,
            src_kv_ptrs=src_kv_blocks_transfer_ptrs,
            dst_kv_ptrs=dst_kv_blocks_transfer_ptrs,
            kv_sizes=[src_kv_blocks_size] * len(src_kv_blocks_transfer_ptrs),
            expect_count=expect_count,
            remote_name=peer_instance_rank_info.instance_name +
            str(peer_instance_rank_info.instance_rank),
            peer_endpoint=peer_instance_rank_info.recv_endpoint,
            peer_session_id=dst_info.session_id)

    def get_state(self) -> SessionState:
        pass

    def trigger_send_chunk(self, chunk_id, chunk_num):
        # call sender_submit_send_task
        pass

    def is_active(self) -> bool:
        if self.first_extracted:
            return self.encountered_count < self.expect_count
        else:
            return True

    def _need_send_transfer(self, peer_domain_targets: peerDomainTargets,
                            peer_instance_rank_info: InstanceRankInfo) -> bool:
        if peer_domain_targets.duplicate_head_factor <= 1:
            return True
        peer_dp_rank = peer_instance_rank_info.dp_rank if peer_instance_rank_info.enable_attention_dp else 0
        self_tp_size_per_dp_group = self.resource_register.get_instance_rank_info(
        ).tp_size // self.resource_register.get_instance_rank_info(
        ).dp_size if self.resource_register.get_instance_rank_info(
        ).enable_attention_dp else self.resource_register.get_instance_rank_info(
        ).tp_size
        self_tprank_in_dp_group = self.resource_register.get_instance_rank_info(
        ).tp_rank % self_tp_size_per_dp_group
        return (peer_dp_rank % peer_domain_targets.duplicate_head_factor) == (
            self_tprank_in_dp_group % peer_domain_targets.duplicate_head_factor)

    def _filter_kv_blocks(self, src_block_ids,
                          dst_block_ids) -> tuple[list[int], list[int]]:

        # TODO: filter the kv blocks according to the peer_domain_targets
        return src_block_ids, dst_block_ids


class ReceiverSession(TransferSession):

    def __init__(self, kv_cache_manager: KVCacheManager, request_id: int,
                 disagg_params: DisaggregatedParams,
                 resource_register: ResourceRegistrar):
        self.disagg_params = disagg_params
        super().__init__(kv_cache_manager, request_id, resource_register)

    def extract_trans_meta(
            self, peer_instance_info: InstanceInfo,
            peer_dp_rank: int) -> tuple[AgentRecvArgs, list[int]]:
        peer_domain_ranks = self.resource_register.get_peer_domain_targets(
            peer_instance_info, peer_dp_rank)
        expect_count = len(peer_domain_ranks.ranks)
        if not self.first_extracted:
            self.first_extracted = True
            self.expect_count = len(peer_domain_ranks.ranks)
        self.encountered_count = self.encountered_count + 1
        return AgentRecvArgs(session_id=self.session_id,
                             future_for_session=self.future,
                             expect_count=expect_count,
                             remote_name=None), peer_domain_ranks.ranks

    def create_gen_side_transfer_req_info(
            self, disagg_params: DisaggregatedParams) -> TransferGenSideReqInfo:

        # TODO:
        ctx_request_id = disagg_params.ctx_request_id

        block_ids = self.kv_cache_manager.get_batch_cache_indices(
            [self.request_id])[0]

        return TransferGenSideReqInfo(ctx_req_id=ctx_request_id,
                                      instance_name=self.resource_register.
                                      get_instance_rank_info().instance_name,
                                      instance_rank=self.resource_register.
                                      get_instance_rank_info().instance_rank,
                                      block_ids=block_ids,
                                      gen_req_id=self.request_id,
                                      start_token_id=0,
                                      session_id=self.session_id)

    def get_state(self) -> SessionState:
        pass


class Sender:

    def __init__(self, kv_cache_manager: KVCacheManager,
                 resource_register: ResourceRegistrar, device_id: int,
                 transfer_agent: BaseTransferAgent):
        self.kv_cache_manager = kv_cache_manager
        self.resource_register = resource_register
        self.device_id = device_id
        self.transfer_agent = transfer_agent
        self.send_session_cache = {}
        self.send_session_cache_lock = threading.Lock()

        self._peer_transfer_recv_req_info_cache = {}
        self._peer_transfer_recv_req_info_lock = threading.Lock()

        self._zmq_context = zmq.Context()
        self.server_socket = self._zmq_context.socket(zmq.ROUTER)
        self.server_socket.bind(f"tcp://{get_local_ip()}:*")
        self.server_endpoint = self.server_socket.getsockopt(
            zmq.LAST_ENDPOINT).decode()

        self.socket_cache = {}

        self.session_id_to_count = {}
        print(f" Sender init end with server_endpoint: {self.server_endpoint}")

        background_thread = threading.Thread(target=self._handle_sender_loop,
                                             daemon=True)
        background_thread.start()

    def get_endpoint(self):
        return self.server_endpoint

    #  return a session for a request.
    #  upper layer can check the state of the session to know the progress of the request.
    #
    def async_send(self, request: LlmRequest) -> SenderSession:
        request_id = request.py_request_id

        if request_id in self.send_session_cache:
            return self.send_session_cache[request_id]
        send_session = SenderSession(self.kv_cache_manager, request_id,
                                     self.resource_register)
        self.send_session_cache[request_id] = send_session
        print(f" call _handle_send_session send_session: {send_session}")
        self._handle_send_session(send_session)
        return send_session

    #  for upper layer to create a send session for a request ,only used for pre-allocate flow
    def create_send_session(self, request: LlmRequest) -> SenderSession:

        #  for upper layer to create a send session for a request ,only used for pre-allocate flow. the kvcache will not send until session.trigger_send_chunk() is called.

        pass

    def cancel_request(self, request: LlmRequest):
        pass

    def submit_send_task(self, trans_send_meta: AgentSendArgs):
        # print(f" call submit_send_task trans_send_meta: {trans_send_meta}")
        if not hasattr(self, '_send_task_queue'):
            import queue
            import threading
            self._send_task_queue = queue.Queue()
            self._background_thread = threading.Thread(
                target=self._handle_send_task_loop, daemon=True)
            self._background_thread.start()
        self._send_task_queue.put(trans_send_meta)

    def _handle_send_task_loop(self):
        while True:
            agent_send_args = self._send_task_queue.get()
            if agent_send_args is None:
                break
            self._handle_send_task(agent_send_args)

    def _handle_send_task(self, agent_send_args: AgentSendArgs):
        assert len(agent_send_args.src_kv_ptrs) == len(
            agent_send_args.dst_kv_ptrs)
        assert len(agent_send_args.kv_sizes) == len(agent_send_args.src_kv_ptrs)
        src_kv_list = [(src_ptr, size, self.device_id) for src_ptr, size in zip(
            agent_send_args.src_kv_ptrs, agent_send_args.kv_sizes)]
        dst_kv_list = [(dst_ptr, size, self.device_id) for dst_ptr, size in zip(
            agent_send_args.dst_kv_ptrs, agent_send_args.kv_sizes)]
        if agent_send_args.session_id not in self.session_id_to_count:
            self.session_id_to_count[agent_send_args.session_id] = 0
        src_memory_descs = MemoryDescs("VRAM", src_kv_list)
        dst_memory_descs = MemoryDescs("VRAM", dst_kv_list)
        request = TransferRequest(TransferOp.WRITE, src_memory_descs,
                                  dst_memory_descs, agent_send_args.remote_name,
                                  '')
        status = self.transfer_agent.submit_transfer_requests(request)
        sync_status = "SUCCESS"
        if not status.wait():
            sync_status = "FAILED"
            agent_send_args.future_for_session.set_exception(
                RuntimeError("Transfer failed"))
        socket = self._get_socket(agent_send_args.peer_endpoint)
        socket.send_multipart([
            str(MessageType.TASK_STATE).encode("ascii"),
            agent_send_args.peer_session_id.encode("ascii"),
            sync_status.encode("ascii")
        ])
        self.session_id_to_count[agent_send_args.session_id] += 1
        if (self.session_id_to_count[agent_send_args.session_id]
                > agent_send_args.expect_count):
            agent_send_args.future_for_session.set_exception(
                RuntimeError(
                    f"Session {agent_send_args.session_id} has more than {agent_send_args.expect_count} transfers"
                ))
        elif (self.session_id_to_count[agent_send_args.session_id] ==
              agent_send_args.expect_count):
            agent_send_args.future_for_session.set_result(sync_status)
            del self.session_id_to_count[agent_send_args.session_id]

    def _handle_send_session(self, send_session: SenderSession):
        transfer_recv_req_info_dict = {}
        with self._peer_transfer_recv_req_info_lock:
            if send_session.request_id in self._peer_transfer_recv_req_info_cache:
                transfer_recv_req_info_dict = self._peer_transfer_recv_req_info_cache[
                    send_session.request_id]
        for transfer_recv_req_info in transfer_recv_req_info_dict.values():
            trans_meta = send_session.extract_trans_meta(transfer_recv_req_info)
            # print(f" call submit_send_task trans_meta: {trans_meta}")
            self.submit_send_task(trans_meta)
            if (not send_session.is_active()):
                with self.send_session_cache_lock:
                    if send_session.request_id in self.send_session_cache:
                        del self.send_session_cache[send_session.request_id]

    def _handle_sender_loop(self):
        while True:
            message = self.server_socket.recv_multipart()
            send_id = message[0]
            recv_message = message[1:]
            if self._message_is_termination(recv_message):
                break
            elif self._message_is_request_data(recv_message):
                self._handle_request_data(send_id, recv_message)
            elif self._message_is_request_instance_info(recv_message):
                self._handle_request_instance_info(send_id, recv_message)

            elif self._message_is_register_rank_info(recv_message):
                self._handle_register_rank_info(send_id, recv_message)
            else:
                raise ValueError(
                    f"transceiver sender loop received unknown message type: {recv_message[0]}"
                )

    def _message_is_termination(self, message: list[bytes]):
        return message[0] == str(MessageType.TERMINATION).encode("ascii")

    def _message_is_request_data(self, message: list[bytes]):
        # mapping_info = self._convert_message_to_mapping_info(message)
        return message[0] == str(MessageType.REQUEST_DATA).encode("ascii")

    def _message_is_request_instance_info(self, message: list[bytes]):
        return message[0] == str(
            MessageType.REQUEST_INSTANCE_INFO).encode("ascii")

    def _message_is_register_rank_info(self, message: list[bytes]):
        return message[0] == str(MessageType.REGISTER_RANK_INFO).encode("ascii")

    def _handle_request_instance_info(self, send_id: bytes,
                                      message: list[bytes]):
        assert len(message) == 1
        send_message = [
            send_id,
            pickle.dumps(self.resource_register.get_instance_info())
        ]
        self.server_socket.send_multipart(send_message)

    def _handle_register_rank_info(self, send_id: bytes, message: list[bytes]):
        instance_rank_info: InstanceRankInfo = pickle.loads(message[1])
        # print(
        #     f" _handle_register_rank_info instance_rank_info: {instance_rank_info}"
        # )
        self.resource_register.register_peer_info(
            instance_rank_info.instance_name, instance_rank_info.instance_rank,
            instance_rank_info)
        self.transfer_agent.load_remote_agent(
            instance_rank_info.instance_name +
            str(instance_rank_info.instance_rank),
            instance_rank_info.transfer_engine_info)
        print(
            f" _handle_register_rank_info end with instance_name , instance_rank: {instance_rank_info.instance_name} , {instance_rank_info.instance_rank}"
        )

    def _get_send_transfer_session(self, ctx_req_id: int) -> TransferSession:
        with self.send_session_cache_lock:
            if ctx_req_id in self.send_session_cache:
                return self.send_session_cache[ctx_req_id]
            else:
                return None

    def _handle_request_data(self, send_id: bytes, message: list[bytes]):

        transfer_gen_side_req_info: TransferGenSideReqInfo = pickle.loads(
            message[1])

        ctx_req_id = transfer_gen_side_req_info.ctx_req_id

        send_transfer_session = self._get_send_transfer_session(ctx_req_id)
        if send_transfer_session is None:
            print(f" _handle_request_data, send_transfer_session is None")
            self._save_peer_transfer_req_info(transfer_gen_side_req_info)
        else:
            print(f" _handle_request_data, send_transfer_session is not None")
            trans_meta = send_transfer_session.extra_trans_meta(
                transfer_gen_side_req_info)
            print(f" call submit_send_task trans_meta: {trans_meta}")
            self.submit_send_task(trans_meta)
            #           # do we need big lock to protect is_active(), remain_count
            if (not send_transfer_session.is_active()):
                with self.send_session_cache_lock:
                    print(" handle_request_data, delete send_session_cache")

                    if ctx_req_id in self.send_session_cache:
                        del self.send_session_cache[ctx_req_id]

    def _get_socket(self, endpoint: str):
        if endpoint not in self.socket_cache:
            self.socket_cache[endpoint] = self._zmq_context.socket(zmq.DEALER)
            self.socket_cache[endpoint].connect(endpoint)
        return self.socket_cache[endpoint]

    def _save_peer_transfer_req_info(
            self, peer_transfer_req_info: TransferGenSideReqInfo):
        with self._peer_transfer_recv_req_info_lock:
            if peer_transfer_req_info.ctx_req_id not in self._peer_transfer_recv_req_info_cache:
                self._peer_transfer_recv_req_info_cache[
                    peer_transfer_req_info.ctx_req_id] = {}
            self._peer_transfer_recv_req_info_cache[
                peer_transfer_req_info.ctx_req_id][
                    peer_transfer_req_info.
                    instance_rank] = peer_transfer_req_info


class Receiver:

    def __init__(self, kv_cache_manager: KVCacheManager,
                 resource_register: ResourceRegistrar, device_id: int,
                 transfer_agent: BaseTransferAgent):
        self.kv_cache_manager = kv_cache_manager
        self.resource_register = resource_register
        self.device_id = device_id
        self.transfer_agent = transfer_agent
        self.receive_session_cache = {}
        self.receive_session_cache_lock = threading.Lock()

        self._zmq_context = zmq.Context()
        self.socket_cache = {}
        self.context_endpoint_to_instance_info_cache = {}
        self.request_id_to_receiver_session_cache = {}
        self.request_id_to_receiver_session_cache_lock = threading.Lock()

        self.receiver_server_socket = self._zmq_context.socket(zmq.ROUTER)
        self.receiver_server_socket.bind(f"tcp://{get_local_ip()}:*")
        self.receiver_server_endpoint = self.receiver_server_socket.getsockopt(
            zmq.LAST_ENDPOINT).decode()
        self._receiver_background_thread = threading.Thread(
            target=self._handle_receiver_loop, daemon=True)
        self._receiver_background_thread.start()

        self.session_id_to_future = {}
        self.session_id_to_count = {}
        self.session_id_to_count_lock = threading.Lock()
        self.session_id_to_future_lock = threading.Lock()

        print(
            f" Receiver init end with receiver_server_endpoint: {self.receiver_server_endpoint}"
        )

    def get_endpoint(self):
        return self.receiver_server_endpoint

    def async_receive(self,
                      request: LlmRequest,
                      start_token_id: Optional[int] = None) -> ReceiverSession:
        # for upper layer to create a receive session for a request, and the receiver will async wait the receive finished.

        return self._async_request_data_transfer(request)

    def _async_request_data_transfer(self, request: LlmRequest):
        disagg_params: DisaggregatedParams = request.py_disaggregated_params
        print(f" _async_request_data_transfer disagg_params: {disagg_params}")
        context_peer_infos: InstanceInfo = self._get_context_info(disagg_params)
        print(
            f" _async_request_data_transfer context_peer_infos: {context_peer_infos}"
        )
        receiver_session = ReceiverSession(self.kv_cache_manager,
                                           request.py_request_id, disagg_params,
                                           self.resource_register)
        with self.request_id_to_receiver_session_cache_lock:
            self.request_id_to_receiver_session_cache[
                request.py_request_id] = receiver_session
        print(
            f" _async_request_data_transfer request_id_to_receiver_session_cache: {self.request_id_to_receiver_session_cache}"
        )
        transfer_gen_side_req_info = receiver_session.create_gen_side_transfer_req_info(
            disagg_params)

        agent_recv_args, target_ranks = receiver_session.extract_trans_meta(
            context_peer_infos, disagg_params.ctx_dp_rank)

        for rank in target_ranks:
            self._send_data_request(
                context_peer_infos.ctx_server_endpoints[rank],
                transfer_gen_side_req_info)
        self.submit_receive_task(agent_recv_args)
        return receiver_session

    def _need_register_peer_in_first_request(
            self, disagg_params: DisaggregatedParams) -> bool:
        return disagg_params.ctx_leader_endpoint not in self.context_endpoint_to_instance_info_cache

    def _get_socket(self, endpoint: str):
        if endpoint not in self.socket_cache:
            self.socket_cache[endpoint] = self._zmq_context.socket(zmq.DEALER)
            self.socket_cache[endpoint].connect(endpoint)
        return self.socket_cache[endpoint]

    def _get_context_info(self,
                          disagg_params: DisaggregatedParams) -> InstanceInfo:
        if self._need_register_peer_in_first_request(disagg_params):
            socket = self._zmq_context.socket(zmq.DEALER)
            socket.connect(disagg_params.ctx_leader_endpoint)
            message = [str(MessageType.REQUEST_INSTANCE_INFO).encode("ascii")]
            socket.send_multipart(message)
            message = socket.recv_multipart()
            instance_info = pickle.loads(message[0])
            print(f" _get_context_info instance_info: {instance_info}")
            socket.close()

            for endpoint in instance_info.ctx_server_endpoints:
                socket = self._get_socket(endpoint)
                send_message = []
                send_message.append(
                    str(MessageType.REGISTER_RANK_INFO).encode("ascii"))
                send_message.append(
                    pickle.dumps(
                        self.resource_register.get_instance_rank_info()))
                socket.send_multipart(send_message)

            self.context_endpoint_to_instance_info_cache[
                disagg_params.ctx_leader_endpoint] = instance_info
            return instance_info

        else:
            return self.context_endpoint_to_instance_info_cache[
                disagg_params.ctx_leader_endpoint]

    def submit_receive_task(self, trans_recv_meta: AgentRecvArgs):
        print(f" call submit_receive_task trans_recv_meta: {trans_recv_meta}")
        if trans_recv_meta.session_id not in self.session_id_to_count:
            self.session_id_to_count[
                trans_recv_meta.session_id] = trans_recv_meta.expect_count
            self.session_id_to_future[
                trans_recv_meta.session_id] = trans_recv_meta.future_for_session
        # async wait the write finished signal from sender
        else:
            assert self.session_id_to_future[
                trans_recv_meta.
                session_id] == trans_recv_meta.future_for_session

    def _handle_receiver_loop(self):

        while True:
            message = self.receiver_server_socket.recv_multipart()
            send_id = message[0]
            recv_message = message[1:]
            if self._message_is_termination(recv_message):
                break
            elif self._message_is_task_state(recv_message):
                self._handle_task_state(send_id, recv_message)
            else:
                raise ValueError(
                    f"transceiver receiver loop received unknown message type: {recv_message[0]}"
                )

    def _message_is_termination(self, message: list[bytes]):
        return message[0] == str(MessageType.TERMINATION).encode("ascii")

    def _message_is_task_state(self, message: list[bytes]):
        return message[0] == str(MessageType.TASK_STATE).encode("ascii")

    def _handle_task_state(self, send_id: bytes, message: list[bytes]):
        assert len(message) == 3
        assert message[0].decode("ascii") == str(MessageType.TASK_STATE)
        session_id = message[1].decode("ascii")
        task_state = message[2].decode("ascii")
        if task_state == "SUCCESS":
            self.session_id_to_count[session_id] -= 1
            if self.session_id_to_count[session_id] == 0:
                self.session_id_to_future[session_id].set_result("SUCCESS")
                self.session_id_to_future.pop(session_id)
                del self.session_id_to_count[session_id]
        elif task_state == "FAILED":
            self.session_id_to_future[session_id].set_exception(
                RuntimeError(f"Task state: {task_state}"))
        else:
            raise ValueError(
                f" session {session_id} received unknown task state: {task_state}"
            )

    def _send_data_request(self, endpoint: str,
                           transfer_gen_side_req_info: TransferGenSideReqInfo):
        print(
            f" call _send_data_request endpoint: {endpoint} transfer_gen_side_req_info: {transfer_gen_side_req_info}"
        )
        socket = self._get_socket(endpoint)
        send_message = []
        send_message.append(str(MessageType.REQUEST_DATA).encode("ascii"))
        send_message.append(pickle.dumps(transfer_gen_side_req_info))
        socket.send_multipart(send_message)


class TransferAgentConfig:
    pass


class TransferWorker:

    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        mapping: Mapping,
        device_id: int,
        instance_name: str,
        transfer_agent_config: TransferAgentConfig,
    ):

        self.mapping = mapping

        self.instance_info: InstanceInfo = None
        self.instance_rank_info: InstanceRankInfo = None
        self.kv_cache_manager = kv_cache_manager
        self.init_instance_info(instance_name)

        self.resource_register = ResourceRegistrar(self.instance_rank_info,
                                                   self.instance_info)
        self.device_id = device_id
        self.transfer_agent = NixlTransferAgent(
            self.instance_rank_info.instance_name +
            str(self.instance_rank_info.instance_rank), True)

        self._register_kv_cache()
        self.sender = Sender(kv_cache_manager, self.resource_register,
                             device_id, self.transfer_agent)
        self.receiver = Receiver(kv_cache_manager, self.resource_register,
                                 device_id, self.transfer_agent)
        self.instance_rank_info.transfer_engine_info = bytes(
            self.transfer_agent.get_local_agent_desc())
        self.instance_rank_info.server_endpoint = self.sender.get_endpoint()
        self.instance_rank_info.recv_endpoint = self.receiver.get_endpoint()

    def update_instance_info_with_collective_info(
            self, update_endpoints: list[str],
            update_layer_num_per_pp: list[int]):
        self.instance_info.ctx_server_endpoints = update_endpoints
        self.instance_info.layer_num_per_pp = update_layer_num_per_pp
        self.instance_rank_info.layer_num_per_pp = update_layer_num_per_pp

    def create_send_session(self, request: LlmRequest) -> SenderSession:
        return self.sender.create_send_session(request)

    def async_send(self, request: LlmRequest) -> SenderSession:
        return self.sender.async_send(request)

    def async_receive(self, request: LlmRequest) -> ReceiverSession:
        return self.receiver.async_receive(request)

    def cancel_request(self, request: LlmRequest):
        pass

    def init_instance_info(self, instance_name):
        rank = self.mapping.rank

        tp_size = self.mapping.tp_size
        pp_size = self.mapping.pp_size
        dp_size = self.mapping.dp_size
        cp_size = self.mapping.cp_size
        tp_rank = self.mapping.tp_rank
        pp_rank = self.mapping.pp_rank
        enable_attention_dp = self.mapping.enable_attention_dp
        dp_rank = 0
        if (enable_attention_dp):
            dp_size = self.mapping.tp_size
            dp_rank = tp_rank
        cp_rank = self.mapping.cp_rank
        is_mla = self.kv_cache_manager.kv_factor == 1
        self.kv_cache_manager.kv_factor
        heads_num_per_rank = self.kv_cache_manager.num_kv_heads
        tokens_per_block = self.kv_cache_manager.tokens_per_block
        dims_per_head = self.kv_cache_manager.head_dim
        element_size = get_size_in_bytes(1, self.kv_cache_manager.dtype)
        layer_num_per_pp = [self.kv_cache_manager.num_layers]
        ctx_server_endpoints = []
        self.instance_info = InstanceInfo(
            instance_name=instance_name,
            tp_size=tp_size,
            pp_size=pp_size,
            dp_size=dp_size,
            cp_size=cp_size,
            kv_head_num_per_rank=heads_num_per_rank,
            tokens_per_block=tokens_per_block,
            dims_per_head=dims_per_head,
            element_size=element_size,
            enable_attention_dp=enable_attention_dp,
            is_mla=is_mla,
            layer_num_per_pp=layer_num_per_pp,
            ctx_server_endpoints=ctx_server_endpoints)
        self.instance_rank_info = InstanceRankInfo(
            instance_name=instance_name,
            instance_rank=rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            pp_size=pp_size,
            pp_rank=pp_rank,
            dp_size=dp_size,
            dp_rank=dp_rank,
            cp_size=cp_size,
            cp_rank=cp_rank,
            kv_head_num_per_rank=heads_num_per_rank,
            tokens_per_block=tokens_per_block,
            dims_per_head=dims_per_head,
            element_size=element_size,
            enable_attention_dp=enable_attention_dp,
            is_mla=is_mla,
            layer_num_per_pp=layer_num_per_pp,
            kvcache_ptrs=[
                self.kv_cache_manager.get_unique_primary_pool().data_ptr()
            ],
            aux_ptrs=[],
            server_endpoint="",
            recv_endpoint="",
            transfer_engine_info=bytes())

    def _register_kv_cache(self):
        memory_pool = self.kv_cache_manager.get_unique_primary_pool()
        memory_desc = (memory_pool.data_ptr(),
                       memory_pool.numel() * memory_pool.element_size(),
                       self.device_id, "kv_cache_memory")
        reg_memory_desc = RegMemoryDescs("VRAM", [memory_desc])
        self.transfer_agent.register_memory(reg_memory_desc)


def test_transfer_worker():
    mapping = Mapping(world_size=1, rank=0)
    num_layers = 2
    head_dim = 128
    num_kv_heads = 4
    tokens_per_block = 8
    max_seq_len = 256
    max_batch_size = 1
    dtype = DataType.FLOAT

    ctx_kv_cache_manager = KVCacheManager(
        trtllm.KvCacheConfig(
            max_tokens=2048,
            enable_block_reuse=False,
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=mapping,
        dtype=dtype)

    ctx_instance_name = "ctx_instance"
    transfer_agent_config = TransferAgentConfig()
    ctx_transfer_worker = TransferWorker(
        kv_cache_manager=ctx_kv_cache_manager,
        mapping=mapping,
        device_id=0,
        instance_name=ctx_instance_name,
        transfer_agent_config=transfer_agent_config)
    ctx_enpoint = ctx_transfer_worker.sender.server_endpoint
    ctx_layer_num_per_pp = [num_layers]
    ctx_transfer_worker.update_instance_info_with_collective_info(
        update_endpoints=[ctx_enpoint],
        update_layer_num_per_pp=ctx_layer_num_per_pp)
    gen_kv_cache_manager = KVCacheManager(
        trtllm.KvCacheConfig(
            max_tokens=2048,
            enable_block_reuse=False,
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=mapping,
        dtype=dtype)

    gen_instance_name = "gen_instance"

    gen_transfer_worker = TransferWorker(
        kv_cache_manager=gen_kv_cache_manager,
        mapping=mapping,
        device_id=0,
        instance_name=gen_instance_name,
        transfer_agent_config=transfer_agent_config)

    gen_enpoint = gen_transfer_worker.sender.server_endpoint
    gen_layer_num_per_pp = [num_layers]
    gen_transfer_worker.update_instance_info_with_collective_info(
        update_endpoints=[gen_enpoint],
        update_layer_num_per_pp=gen_layer_num_per_pp)

    sampling_params = SamplingParams()

    ctx_request = LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)

    ctx_kv_cache_manager.impl.add_sequence(ctx_request.py_request_id,
                                           ctx_request.prompt_len, 1,
                                           ctx_request)
    ctx_block_ids = ctx_kv_cache_manager.get_batch_cache_indices(
        [ctx_request.py_request_id])[0]

    ctx_block_data_pools = ctx_kv_cache_manager.get_unique_primary_pool()

    random_values = torch.rand(ctx_block_data_pools.shape,
                               dtype=torch.float32,
                               device=ctx_block_data_pools.device)
    ctx_block_data_pools.copy_(random_values)

    gen_request = LlmRequest(
        request_id=1,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY)

    gen_request.py_disaggregated_params = DisaggregatedParams(
        ctx_request_id=ctx_request.py_request_id,
        ctx_dp_rank=0,
        ctx_leader_endpoint=ctx_transfer_worker.instance_info.
        ctx_server_endpoints[0])
    gen_kv_cache_manager.impl.add_sequence(gen_request.py_request_id,
                                           gen_request.prompt_len, 1,
                                           gen_request)
    print(f"gen async_receive before")
    recv_session = gen_transfer_worker.async_receive(gen_request)

    time.sleep(0.1)
    print(f"gen async_receive after")
    print(f"ctx async_send before")
    send_session = ctx_transfer_worker.async_send(ctx_request)
    print(f"ctx async_send after")

    print(
        f"send_session.get_future_for_session().result(): {send_session.get_future_for_session().result()}"
    )
    print(
        f"recv_session.get_future_for_session().result(): {recv_session.get_future_for_session().result()}"
    )
    gen_block_ids = gen_kv_cache_manager.get_batch_cache_indices(
        [gen_request.py_request_id])[0]
    print(f"gen_block_ids: {gen_block_ids}")
    gen_block_datas = gen_kv_cache_manager.get_unique_primary_pool(
    )[gen_block_ids]

    ctx_block_datas = ctx_kv_cache_manager.get_unique_primary_pool(
    )[ctx_block_ids]

    print(
        f"ctx_block_datas: {ctx_block_datas}, ctx_block_datas.shape: {ctx_block_datas.shape}, ctx_block_datas.data_ptr: {ctx_block_datas.data_ptr()}"
    )
    print(
        f"gen_block_datas: {gen_block_datas}, gen_block_datas.shape: {gen_block_datas.shape}, gen_block_datas.data_ptr: {gen_block_datas.data_ptr()}"
    )
    assert ctx_block_datas.equal(gen_block_datas)


if __name__ == "__main__":
    test_transfer_worker()

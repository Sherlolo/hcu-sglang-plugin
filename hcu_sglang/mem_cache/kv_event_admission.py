from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

from sglang.srt.runtime_context import get_lora
from hcu_sglang.mem_cache.kv_events import KvEventLoraNames
from hcu_sglang.utils.detection import is_hcu

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

_ADMISSION = ContextVar("hcu_kv_event_admission", default=None)


def init_kv_event_lora_names(self) -> None:
    """Init the table that maps lora_id to adapter name for KV events."""
    self.kv_event_lora_names = KvEventLoraNames.from_lora_refs(
        get_lora().lora_paths
    )

def _record_kv_event_lora_name(self, req: Req, lora_name: Optional[str]) -> None:
    # Keyed by the final extra_key, so it must run after elastic namespacing.
    # Only a cache whose events are taken prunes the table.
    if (
        self.kv_events_publisher.enable_kv_cache_events
        and self.tree_cache.supports_prefix_sharing()
    ):
        self.kv_event_lora_names.register(req, lora_name)



def around_init(original, self):
    if not is_hcu():
        return original(self)
    return init_kv_event_lora_names(self)


def around_admission(original, self, recv_req, *args, **kwargs):
    if not is_hcu():
        return original(self, recv_req, *args, **kwargs)
    token = _ADMISSION.set((self, recv_req))
    try:
        return original(self, recv_req, *args, **kwargs)
    finally:
        _ADMISSION.reset(token)


def around_namespace(original, self, req):
    result = original(self, req)
    admission = _ADMISSION.get()
    if is_hcu() and admission is not None and admission[0] is self:
        _record_kv_event_lora_name(self, req, getattr(admission[1], "lora_name", None))
    return result


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    root = "sglang.srt.managers.scheduler.Scheduler."
    HookRegistry.register(root + "init_kv_event_lora_names", around_init, HookType.AROUND)
    HookRegistry.register(root + "handle_generate_request", around_admission, HookType.AROUND)
    HookRegistry.register(root + "handle_embedding_request", around_admission, HookType.AROUND)
    HookRegistry.register(root + "_maybe_namespace_elastic_radix_cache", around_namespace, HookType.AROUND)

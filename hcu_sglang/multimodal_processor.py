"""Compatibility hooks for the generic Transformers multimodal processor."""

from __future__ import annotations

from typing import Any, Callable


def after_build_mm_items(
    items: list[Any], processor: Any, processor_output: dict, input_ids: Any
) -> list[Any]:
    del processor_output
    from sglang.srt.managers.schedule_batch import Modality

    multi_images = getattr(Modality, "MULTI_IMAGES", None)
    if multi_images is None:
        return items

    image_token_id = processor.mm_tokens.image_token_id
    if image_token_id is None:
        return items
    for item in items:
        if item.modality == multi_images:
            item.offsets = processor.get_mm_items_offset(input_ids, image_token_id)
    return items


async def around_process_mm_data_async(
    original: Callable[..., Any], *args, **kwargs
) -> Any:
    output = await original(*args, **kwargs)
    if output is None:
        return None

    token_type_ids = getattr(output, "token_type_ids", None)
    if token_type_ids is not None and not isinstance(token_type_ids, list):
        tolist = getattr(token_type_ids, "tolist", None)
        if tolist is not None:
            output.token_type_ids = tolist()
    return output


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    target = (
        "sglang.srt.multimodal.processors.transformers_auto."
        "TransformersAutoMultimodalProcessor"
    )
    HookRegistry.register(
        f"{target}._build_mm_items", after_build_mm_items, HookType.AFTER
    )
    HookRegistry.register(
        f"{target}.process_mm_data_async",
        around_process_mm_data_async,
        HookType.AROUND,
    )

"""HCU GLM5-Next attention and cache contracts migrated from sglang_glm_dev."""


def is_glm5_next_hcu(config) -> bool:
    from hcu_sglang.utils.detection import is_dcu as is_hcu

    config = getattr(config, "text_config", config)
    return is_hcu() and getattr(config, "model_type", "") in (
        "glm5_next",
        "glm5v_next",
        "glm5next_text",
        "glm5_next_text",
    )

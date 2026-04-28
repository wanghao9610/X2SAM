from functools import partial
from os import getenv

import vlmeval.config as vlmeval_config

from .x2sam_chat import X2SamChat

init_dir = getenv("INIT_DIR", "./inits/")

x2sam_series = {
    "llava-phi3-siglip2-ft": partial(
        X2SamChat,
        x2sam_path=init_dir + "llava-phi3-siglip2-ft",
        vision_encoder_path=init_dir + "siglip2-so400m-patch14-384",
        visual_select_layer=-2,
        visual_select_indx=0,
        prompt_template="phi3_chat",
    ),
    "x2sam-phi3-siglip2-sam-l-mft": partial(
        X2SamChat,
        x2sam_path=init_dir + "x2sam-phi3-siglip2-sam-l-mft",
        visual_select_layer=-2,
        visual_select_indx=0,
        prompt_template="phi3_chat",
    ),
    "x2sam-qwen3vl-4b-sam2-hiera-large-mft": partial(
        X2SamChat,
        x2sam_path=init_dir + "x2sam-qwen3vl-4b-sam2-hiera-large-mft",
        visual_select_layer=-2,
        visual_select_indx=0,
        prompt_template="qwen3_instruct",
        expand2square=False,
        use_dual_encoder=False,
        image_token="<|vision_start|><image><|vision_end|>",
    ),
    "x2sam-qwen3vl-4b-sam2-hiera-large-lora": partial(
        X2SamChat,
        x2sam_path=init_dir + "x2sam-qwen3vl-4b-sam2-hiera-large-lora",
        vlm_path=init_dir + "Qwen3-VL-4B-Instruct",
        visual_select_layer=-2,
        visual_select_indx=0,
        prompt_template="qwen3_instruct",
        expand2square=False,
        use_placeholder=True,
        use_dual_encoder=False,
        image_token="<|vision_start|><image><|vision_end|>",
        video_token="<|vision_start|><video><|vision_end|>",
    ),
}

vlmeval_config.supported_VLM.update(x2sam_series)
supported_VLM = vlmeval_config.supported_VLM

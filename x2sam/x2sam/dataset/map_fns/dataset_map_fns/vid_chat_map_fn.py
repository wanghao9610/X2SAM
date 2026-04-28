from x2sam.utils.constants import DEFAULT_VIDEO_TOKEN


def vid_chat_map_fn(
    example,
    video_token=None,
    **kwargs,
):
    messages = example["conversations"]
    input = ""
    conversations = []
    while messages and messages[0]["from"] == "gpt":
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg["from"] == "human":
            if DEFAULT_VIDEO_TOKEN in msg["value"]:
                msg["value"] = msg["value"].replace(DEFAULT_VIDEO_TOKEN, "").strip()
                msg["value"] = (video_token or f"{DEFAULT_VIDEO_TOKEN}\n") + msg["value"]
                msg["value"] = msg["value"].strip()
            input += msg["value"]
        elif msg["from"] == "gpt":
            conversations.append({"input": input, "output": msg["value"]})
            input = ""
        else:
            raise NotImplementedError

    example.update({"conversations": conversations})
    return example

import random

from x2sam.utils.constants import DEFAULT_SEG_TOKEN, DEFAULT_VIDEO_TOKEN

SEG_QUESTIONS = [
    "Could you please give me a brief description of the video? Please respond with interleaved segmentation masks for the corresponding parts of the answer.",
    "Can you provide a brief description of this video? Please output interleaved segmentation masks for the corresponding phrases.",
    "Please briefly describe the contents of the video. Please respond with interleaved segmentation masks for the corresponding parts of the answer.",
    "Could you give a brief explanation of what can be found within this video? Please output interleaved segmentation masks for the corresponding phrases.",
    "Could you give me an brief explanation of this video? Please respond with interleaved segmentation masks for the corresponding phrases.",
    "Could you provide me with a briefly analysis of this video? Please output interleaved segmentation masks for the corresponding parts of the answer.",
]

ANSWER_LIST = [
    f"{DEFAULT_SEG_TOKEN}.",
    f"And {DEFAULT_SEG_TOKEN}.",
    f"The segmentation mask is {DEFAULT_SEG_TOKEN}.",
    f"And the segmentation mask is {DEFAULT_SEG_TOKEN}.",
]


def gcg_seg_conversations(caption, output_ids_with_output=True, cond_type="phrase"):
    questions = []
    answers = []

    question = random.choice(SEG_QUESTIONS)
    if caption is not None:
        caption = caption.strip() + "." if not caption.strip().endswith(".") else caption.strip()
    if output_ids_with_output:
        answer = caption + " " + random.choice(ANSWER_LIST)
    else:
        answer = ""
    questions.append(question)
    answers.append(answer)

    rets = []
    for i, (question, answer) in enumerate(zip(questions, answers)):
        if i == 0:
            rets.append({"from": "human", "value": DEFAULT_VIDEO_TOKEN + question})
        else:
            rets.append({"from": "human", "value": question})
        rets.append({"from": "gpt", "value": answer})
    return rets


def vid_gcgseg_map_fn(
    example,
    output_ids_with_output=True,
    cond_type="phrase",
    video_token=None,
):
    messages = gcg_seg_conversations(example.get("caption", None), output_ids_with_output, cond_type)
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

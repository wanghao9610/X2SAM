from .vid_genseg_evaluator import VidGenSegEvaluator


class VidOVSegEvaluator(VidGenSegEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

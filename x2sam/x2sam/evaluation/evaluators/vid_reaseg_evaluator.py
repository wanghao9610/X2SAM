from .vid_objseg_evaluator import VidObjSegEvaluator


class VidReaSegEvaluator(VidObjSegEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

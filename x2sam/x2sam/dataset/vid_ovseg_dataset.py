from .vid_genseg_dataset import VidGenSegDataset


class VidOVSegDataset(VidGenSegDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

from .vid_genseg_process_fn import vid_genseg_postprocess_fn


def vid_ovseg_postprocess_fn(*args, **kwargs):
    return vid_genseg_postprocess_fn(*args, **kwargs)

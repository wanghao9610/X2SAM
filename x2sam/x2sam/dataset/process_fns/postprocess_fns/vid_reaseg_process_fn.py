from .vid_refseg_process_fn import vid_refseg_postprocess_fn


def vid_reaseg_postprocess_fn(*args, **kwargs):
    return vid_refseg_postprocess_fn(*args, **kwargs)

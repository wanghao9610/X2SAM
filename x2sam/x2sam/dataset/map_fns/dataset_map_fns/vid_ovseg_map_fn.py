from .vid_genseg_map_fn import vid_genseg_map_fn


def vid_ovseg_map_fn(
    *args,
    **kwargs,
):
    return vid_genseg_map_fn(*args, **kwargs)

def __getattr__(name):
    if name == "ExpertOffloadManager":
        from .expert_offload_manager import ExpertOffloadManager

        return ExpertOffloadManager

    # expose the decode-statistics collector on the package, mirroring the
    # lazy-import style above so importing the package still does not pull in
    # torch_npu / the manager.
    if name == "get_decode_stats":
        from .decode_stats import get_decode_stats

        return get_decode_stats
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ExpertOffloadManager", "get_decode_stats"]

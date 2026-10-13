from .joint_hv_env import JointHVEnv, SimBackend

__all__ = ["JointHVEnv", "SimBackend", "RealClusterBackend"]


def __getattr__(name):  # lazy: importing RealClusterBackend pulls in kubectl deps
    if name == "RealClusterBackend":
        from .real_backend import RealClusterBackend
        return RealClusterBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

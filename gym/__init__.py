"""Proof-Gated Adaptive Agent Gym reference implementation."""

from .core import (
    AdmissionReceipt,
    AgentTaskView,
    EpisodeResult,
    HostActionRejected,
    HostTaskDescriptor,
    Observation,
    PolicyDecision,
    PublicTask,
    ReferenceGym,
    TerminalReason,
)
from .host_boundary import HostGateway, HostSessionStart, LocalReferenceHost
from .host_wire import (
    HostProtocolClosed,
    HostProtocolError,
    HostProtocolTimeout,
)
from .ledger import LedgerEntry, TrajectoryLedger
from .private_pack import PrivateHoldoutPack, PrivatePackError, PrivatePackTask
from .subprocess_host import SubprocessHostGateway

__all__ = [
    "AdmissionReceipt",
    "AgentTaskView",
    "EpisodeResult",
    "HostActionRejected",
    "HostGateway",
    "HostProtocolClosed",
    "HostProtocolError",
    "HostProtocolTimeout",
    "HostSessionStart",
    "HostTaskDescriptor",
    "LedgerEntry",
    "LocalReferenceHost",
    "Observation",
    "PolicyDecision",
    "PrivateHoldoutPack",
    "PrivatePackError",
    "PrivatePackTask",
    "PublicTask",
    "ReferenceGym",
    "SubprocessHostGateway",
    "TerminalReason",
    "TrajectoryLedger",
]

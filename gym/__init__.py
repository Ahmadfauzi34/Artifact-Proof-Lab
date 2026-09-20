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
from .ledger import LedgerEntry, TrajectoryLedger

__all__ = [
    "AdmissionReceipt",
    "AgentTaskView",
    "EpisodeResult",
    "HostActionRejected",
    "HostGateway",
    "HostSessionStart",
    "HostTaskDescriptor",
    "LedgerEntry",
    "LocalReferenceHost",
    "Observation",
    "PolicyDecision",
    "PublicTask",
    "ReferenceGym",
    "TerminalReason",
    "TrajectoryLedger",
]

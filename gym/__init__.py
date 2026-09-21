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
from .agent_endpoint import (
    AgentDecisionReceipt,
    AgentEndpoint,
    AgentGenerationReceipt,
    AgentSessionStart,
    LocalReferenceAgentEndpoint,
    SubprocessAgentEndpoint,
)
from .agent_transcript import (
    AgentBoundaryTranscript,
    AgentTranscriptError,
)
from .agent_wire import (
    AgentProtocolClosed,
    AgentProtocolError,
    AgentProtocolTimeout,
)
from .isolated_agent import (
    AgentEndpointPolicyAdapter,
    IsolatedEpisodeResult,
    run_isolated_episode,
)
from .subprocess_host import SubprocessHostGateway

__all__ = [
    "AdmissionReceipt",
    "AgentBoundaryTranscript",
    "AgentDecisionReceipt",
    "AgentEndpoint",
    "AgentEndpointPolicyAdapter",
    "AgentGenerationReceipt",
    "AgentProtocolClosed",
    "AgentProtocolError",
    "AgentProtocolTimeout",
    "AgentSessionStart",
    "AgentTranscriptError",
    "AgentTaskView",
    "EpisodeResult",
    "HostActionRejected",
    "HostGateway",
    "HostProtocolClosed",
    "HostProtocolError",
    "HostProtocolTimeout",
    "HostSessionStart",
    "HostTaskDescriptor",
    "IsolatedEpisodeResult",
    "LedgerEntry",
    "LocalReferenceAgentEndpoint",
    "LocalReferenceHost",
    "Observation",
    "PolicyDecision",
    "PrivateHoldoutPack",
    "PrivatePackError",
    "PrivatePackTask",
    "PublicTask",
    "ReferenceGym",
    "SubprocessAgentEndpoint",
    "SubprocessHostGateway",
    "TerminalReason",
    "TrajectoryLedger",
    "run_isolated_episode",
]

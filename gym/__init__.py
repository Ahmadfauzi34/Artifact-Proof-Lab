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
from .authority_attestation import (
    AuthorityAttestationError,
    AuthorityAttestationReceipt,
    AuthorityAttestationRequest,
    AuthorityClaims,
    AuthorityVerification,
    ReferenceHMACAuthorityAttestor,
    ReferenceHMACAuthorityVerifier,
)
from .attested_external import (
    AttestedExternalEpisodeResult,
    run_attested_external_episode,
)
from .adaptive_policy import (
    CanonicalIntentAdapter,
    LearningReceipt,
    ProofGatedAdaptivePolicy,
    QStat,
    SemanticStateEncoder,
)
from .curriculum import (
    CURRICULUM_FORMAT,
    PROMOTION_RECEIPT_FORMAT,
    Curriculum,
    CurriculumError,
    CurriculumStage,
    PromotionReceipt,
    evaluate_stage,
    load_curriculum,
    validate_curriculum_tasks,
    verify_promotion_chain,
)
from .external_agent import ExternalAgentEndpoint
from .subprocess_host import SubprocessHostGateway

__all__ = [
    "AdmissionReceipt",
    "AttestedExternalEpisodeResult",
    "AuthorityAttestationError",
    "AuthorityAttestationReceipt",
    "AuthorityAttestationRequest",
    "AuthorityClaims",
    "AuthorityVerification",
    "CanonicalIntentAdapter",
    "CURRICULUM_FORMAT",
    "Curriculum",
    "CurriculumError",
    "CurriculumStage",
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
    "ExternalAgentEndpoint",
    "HostActionRejected",
    "HostGateway",
    "HostProtocolClosed",
    "HostProtocolError",
    "HostProtocolTimeout",
    "HostSessionStart",
    "HostTaskDescriptor",
    "IsolatedEpisodeResult",
    "LearningReceipt",
    "LedgerEntry",
    "LocalReferenceAgentEndpoint",
    "LocalReferenceHost",
    "Observation",
    "PROMOTION_RECEIPT_FORMAT",
    "PolicyDecision",
    "PromotionReceipt",
    "ProofGatedAdaptivePolicy",
    "QStat",
    "PrivateHoldoutPack",
    "PrivatePackError",
    "PrivatePackTask",
    "PublicTask",
    "ReferenceGym",
    "ReferenceHMACAuthorityAttestor",
    "ReferenceHMACAuthorityVerifier",
    "SubprocessAgentEndpoint",
    "SemanticStateEncoder",
    "SubprocessHostGateway",
    "TerminalReason",
    "TrajectoryLedger",
    "evaluate_stage",
    "load_curriculum",
    "run_attested_external_episode",
    "run_isolated_episode",
    "validate_curriculum_tasks",
    "verify_promotion_chain",
]

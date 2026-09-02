class ArtifactProofError(Exception):
    """Base class for deterministic artifact rejection."""


class ManifestError(ArtifactProofError):
    """The verification contract is malformed or inconsistent."""


class SourceError(ArtifactProofError):
    """The artifact container is unsafe or unreadable."""

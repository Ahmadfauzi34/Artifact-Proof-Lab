from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass(frozen=True)
class Finding:
    check_id: str
    check_type: str
    status: Status
    message: str
    expected: Any = None
    observed: Any = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return {key: value for key, value in result.items() if value is not None}


@dataclass
class Report:
    artifact: str
    profile: str
    manifest_sha256: str | None = None
    artifact_name: str | None = None
    artifact_version: str | None = None
    findings: list[Finding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(finding.status is Status.FAIL for finding in self.findings)

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def to_dict(self) -> dict[str, Any]:
        counts = {status.value: 0 for status in Status}
        for finding in self.findings:
            counts[finding.status.value] += 1
        result: dict[str, Any] = {
            "format": "artifact-proof-report-v1",
            "status": self.status,
            "artifact": self.artifact,
            "profile": self.profile,
            "summary": counts,
            "findings": [finding.to_dict() for finding in self.findings],
        }
        if self.manifest_sha256 is not None:
            result["manifest_sha256"] = self.manifest_sha256
        if self.artifact_name is not None:
            result["artifact_name"] = self.artifact_name
        if self.artifact_version is not None:
            result["artifact_version"] = self.artifact_version
        return result

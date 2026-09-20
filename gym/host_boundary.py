from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Protocol

from .core import AgentTaskView, Environment, Observation, PublicTask, StepOutcome


@dataclass(frozen=True)
class HostSessionStart:
    """Sanitized session surface returned by a host implementation."""

    session_id: str
    agent_task: AgentTaskView
    observations: tuple[Observation, ...]


class HostGateway(Protocol):
    """Host-side contract.

    A production/private implementation may live in another process or machine.
    The policy never receives PublicTask or the Environment object.
    """

    def start(self, task: PublicTask) -> HostSessionStart: ...
    def step(self, session_id: str, action: str, *, candidate: str | None = None) -> StepOutcome: ...
    def semantic_verdict(self, session_id: str) -> bool: ...
    def close(self, session_id: str) -> None: ...


@dataclass
class _HostSession:
    task: PublicTask
    environment: Environment


class LocalReferenceHost:
    """Inspectable in-repo host used only for reference/regression conformance."""

    def __init__(self, environment_factory: Callable[[PublicTask], Environment]) -> None:
        self._environment_factory = environment_factory
        self._sessions: dict[str, _HostSession] = {}
        self._counter = 0

    def start(self, task: PublicTask) -> HostSessionStart:
        self._counter += 1
        environment = self._environment_factory(task)
        session_id = self._session_id(task, self._counter)
        if session_id in self._sessions:
            raise RuntimeError("host session id collision")
        self._sessions[session_id] = _HostSession(task=task, environment=environment)
        return HostSessionStart(
            session_id=session_id,
            agent_task=task.agent_view(),
            observations=tuple(environment.reset()),
        )

    def step(self, session_id: str, action: str, *, candidate: str | None = None) -> StepOutcome:
        session = self._session(session_id)
        if not _action_allowed(session.task, action):
            raise ValueError(f"action is outside host contract: {action}")
        return session.environment.step(action, candidate=candidate)

    def semantic_verdict(self, session_id: str) -> bool:
        return bool(self._session(session_id).environment.semantic_verdict())

    def close(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _session(self, session_id: str) -> _HostSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown or closed host session: {session_id}") from exc

    @staticmethod
    def _session_id(task: PublicTask, counter: int) -> str:
        material = f"{task.task_id}|{task.environment.get('snapshot_id')}|{counter}".encode("utf-8")
        return "session-" + hashlib.sha256(material).hexdigest()[:20]


def _action_allowed(task: PublicTask, action: str) -> bool:
    allowed = set(task.allowed_actions)
    if action in allowed:
        return True
    if action.startswith("DEFER_") and "DEFER" in allowed:
        return True
    return False

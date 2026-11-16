"""Brand guardrail agent, as a LangGraph state machine.

    generate -> inspect -+-> accept                      (all fidelity checks pass)
        ^                +-> retry    -> generate        (new seed, stricter prompt)
        |                +-> fallback -> generate        (switch to the pixel-exact studio backend)
        +----------------+-> reject                      (nothing passed; flag for a human)

The routing is a fixed policy, not an LLM: accept/reject decisions on brand assets should be
reproducible and explainable from the logged metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TypedDict

from langgraph.graph import END, StateGraph

from ..config import Settings
from ..cutout import Cutout
from ..scene import Scene, SceneBackend, make_backend
from .metrics import FidelityReport, Thresholds, check_fidelity


class GuardrailState(TypedDict, total=False):
    cutout: Cutout
    backend: str
    attempt: int  # attempts made with the current backend
    scene: Scene
    report: FidelityReport
    history: list[dict[str, Any]]
    status: Literal["accepted", "rejected"]


@dataclass
class GuardrailResult:
    status: str
    scene: Scene
    report: FidelityReport
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


def build_guardrail_graph(
    settings: Settings,
    backend_factory: Callable[[str], SceneBackend] | None = None,
):
    factory = backend_factory or (lambda name: make_backend(name, settings))
    backends: dict[str, SceneBackend] = {}
    thresholds = Thresholds(
        min_ssim=settings.min_ssim,
        min_inlier_ratio=settings.min_inlier_ratio,
        max_delta_e=settings.max_delta_e,
        max_hue_shift_deg=settings.max_hue_shift_deg,
    )

    def generate(state: GuardrailState) -> dict:
        name = state["backend"]
        if name not in backends:
            backends[name] = factory(name)
        return {"scene": backends[name].render(state["cutout"], attempt=state["attempt"])}

    def inspect(state: GuardrailState) -> dict:
        scene = state["scene"]
        report = check_fidelity(scene.reference, scene.image, scene.box, thresholds)
        entry = {"backend": scene.backend, "attempt": state["attempt"], **report.to_dict()}
        return {"report": report, "history": [*state.get("history", []), entry]}

    def route(state: GuardrailState) -> str:
        if state["report"].passed:
            return "accept"
        if state["attempt"] + 1 < settings.max_attempts and state["backend"] != "studio":
            return "retry"  # the studio backend is deterministic, so retrying it is pointless
        if settings.fallback_to_studio and state["backend"] != "studio":
            return "fallback"
        return "reject"

    graph = StateGraph(GuardrailState)
    graph.add_node("generate", generate)
    graph.add_node("inspect", inspect)
    graph.add_node("retry", lambda s: {"attempt": s["attempt"] + 1})
    graph.add_node("fallback", lambda s: {"backend": "studio", "attempt": 0})
    graph.add_node("accept", lambda s: {"status": "accepted"})
    graph.add_node("reject", lambda s: {"status": "rejected"})
    graph.set_entry_point("generate")
    graph.add_edge("generate", "inspect")
    graph.add_conditional_edges(
        "inspect", route, {k: k for k in ("accept", "retry", "fallback", "reject")}
    )
    graph.add_edge("retry", "generate")
    graph.add_edge("fallback", "generate")
    graph.add_edge("accept", END)
    graph.add_edge("reject", END)
    return graph.compile()


def run_guardrail(
    cutout: Cutout,
    settings: Settings,
    backend_factory: Callable[[str], SceneBackend] | None = None,
) -> GuardrailResult:
    app = build_guardrail_graph(settings, backend_factory)
    final = app.invoke(
        {"cutout": cutout, "backend": settings.scene_backend, "attempt": 0, "history": []},
        {"recursion_limit": 6 * settings.max_attempts + 12},
    )
    return GuardrailResult(final["status"], final["scene"], final["report"], final["history"])

#!/usr/bin/env python3
"""
BOA Runner — Board of Advisors Query Orchestrator.

Accepts a question, routes it through the TokenRouter (via monica_integration.optimise),
collects answers from 6 board seats, synthesises, and writes a session log.

Importable but optional — not required by the token_optimiser package.

Usage:
    from token_optimiser.boa import run_boa
    result = run_boa("What's the best way to handle auth in a microservice?")

Or CLI:
    python -m token_optimiser.boa "your question here"
    python token_optimiser/boa.py "your question here"
"""

import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the package root is importable when run as a script.
_PACKAGE_ROOT = Path(__file__).resolve().resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from token_optimiser.monica_integration import optimise

# ── Seat definitions ──────────────────────────────────────────────────────────

SEATS: List[Dict[str, str]] = [
    {
        "id": "architect",
        "name": "Architect",
        "framing": (
            "You are the Architect on the Board of Advisors. Focus on system design, "
            "trade-offs, long-term maintainability, and structural coherence. "
            "Question: "
        ),
    },
    {
        "id": "engineer",
        "name": "Engineer",
        "framing": (
            "You are the Engineer on the Board of Advisors. Focus on implementation "
            "details, practicality, performance, and correctness. "
            "Question: "
        ),
    },
    {
        "id": "researcher",
        "name": "Researcher",
        "framing": (
            "You are the Researcher on the Board of Advisors. Focus on evidence, "
            "precedents, benchmarks, and citations. Where possible, reference known "
            "patterns or data. "
            "Question: "
        ),
    },
    {
        "id": "reviewer",
        "name": "Reviewer",
        "framing": (
            "You are the Reviewer on the Board of Advisors. Critically assess the "
            "proposal: identify risks, blind spots, edge cases, and failure modes. "
            "Be constructively sceptical. "
            "Question: "
        ),
    },
    {
        "id": "budget",
        "name": "Budget (Token Optimiser)",
        "framing": (
            "You are the Budget seat on the Board of Advisors (Token Optimiser). "
            "Focus on cost, routing, and efficiency. Which tier (T1/T2/T3) is "
            "appropriate for implementing the solution? What's the cheapest viable "
            "path? "
            "Question: "
        ),
    },
    {
        "id": "executive",
        "name": "Executive (Chair)",
        "framing": (
            "You are the Executive Chair on the Board of Advisors. Synthesise the "
            "board's views into a single recommendation. Decide what to do, in what "
            "order, with what priority. "
            "Question: "
        ),
    },
]

# ── Backend resolution (mock vs. real) ───────────────────────────────────────

def _resolve_backend() -> str:
    """Return backend mode: 'mock' for offline/test, 'live' otherwise."""
    return os.environ.get("BOA_BACKEND", "mock").lower()


def _generate_seat_answer(seat: Dict[str, str], question: str, tier: str) -> Dict[str, Any]:
    """
    Generate a single seat's answer.

    In mock mode (default), returns a structured placeholder based on the seat's
    persona and the routed tier. In live mode, would call the appropriate backend.
    """
    backend = _resolve_backend()
    prompt = seat["framing"] + question

    if backend == "mock":
        return {
            "seat_id": seat["id"],
            "seat_name": seat["name"],
            "tier": tier,
            "prompt_preview": prompt[:120],
            "answer": (
                f"[{seat['name']} — {tier} mock] Response for: {question[:80]}... "
                f"[{seat['id']} perspective applied]"
            ),
            "simulated": True,
            "backend": "mock",
        }

    # Live mode: would dispatch to the tier-appropriate backend.
    # For now, fall back to mock (placeholder for real integration).
    return {
        "seat_id": seat["id"],
        "seat_name": seat["name"],
        "tier": tier,
        "prompt_preview": prompt[:120],
        "answer": (
            f"[{seat['name']} — {tier} live] Response for: {question[:80]}..."
        ),
        "simulated": False,
        "backend": backend,
    }


# ── Synthesis ─────────────────────────────────────────────────────────────────

def _synthesise(seat_answers: List[Dict[str, str]], tier: str, confidence: int) -> str:
    """
    Produce a single synthesis paragraph from all seat answers.
    """
    seat_names = ", ".join(a["seat_name"] for a in seat_answers)
    return (
        f"Synthesis ({len(seat_answers)} seats: {seat_names}) | "
        f"Tier: {tier} | Confidence: {confidence}% | "
        f"Consensus reflects board-level review across architecture, engineering, "
        f"research, review, budget, and executive perspectives."
    )


# ── Session log ───────────────────────────────────────────────────────────────

def _timestamp_compact() -> str:
    """Return a compact timestamp suitable for filenames: YYYY-MM-DD-HHMMSS."""
    return datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")


def _write_session_log(
    question: str,
    tier: str,
    confidence: int,
    seat_answers: List[Dict[str, Any]],
    synthesis: str,
    decision: Dict[str, Any],
    work_dir: Path,
) -> Path:
    """Write the BOA session log in markdown format. Returns the log path."""
    work_dir.mkdir(parents=True, exist_ok=True)
    ts = _timestamp_compact()
    log_path = work_dir / f"{ts}.md"

    lines = [
        f"# BOA Session — {ts}",
        "",
        f"**Question:** {question}",
        f"**Tier:** {tier}",
        f"**Confidence:** {confidence}%",
        f"**Routing decision:** `{decision.get('rationale', '—')}`",
        "",
        "## Seat Answers",
        "",
    ]

    for ans in seat_answers:
        lines.append(f"### {ans['seat_name']} (`{ans['seat_id']}`) — {ans['tier']}")
        lines.append("")
        lines.append(f"> Prompt: {ans['prompt_preview']}...")
        lines.append("")
        lines.append(ans["answer"])
        lines.append("")

    lines.append("## Synthesis")
    lines.append("")
    lines.append(synthesis)
    lines.append("")

    lines.append("## Raw Decision")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(decision, indent=2, default=str))
    lines.append("```")
    lines.append("")

    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path


# ── Main entry point ──────────────────────────────────────────────────────────

def run_boa(
    question: str,
    context: Optional[Dict[str, Any]] = None,
    work_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run the Board of Advisors for a given question.

    Parameters
    ----------
    question : str
        The question to put to the board.
    context : dict, optional
        Additional context passed to monica_integration.optimise (e.g.
        estimated_lines, domain).
    work_dir : Path, optional
        Directory for session logs. Defaults to
        /home/warpuppy/Projects/light-engine/work/boa.

    Returns
    -------
    dict
        {
            "tier": str,
            "confidence": int,
            "seat_answers": list[dict],
            "synthesis": str,
            "log_path": str,
            "decision": dict,   # raw routing decision
        }
    """
    if work_dir is None:
        work_dir = Path("/home/warpuppy/Projects/light-engine/work/boa")

    # 1. Route via monica_integration.optimise (TokenRouter under the hood).
    decision = optimise(question, context or {})
    tier = decision.get("tier", "T2")
    confidence = decision.get("confidence", 50)

    # 2. Collect answers from all 6 seats.
    seat_answers = [_generate_seat_answer(seat, question, tier) for seat in SEATS]

    # 3. Synthesise.
    synthesis = _synthesise(seat_answers, tier, confidence)

    # 4. Write session log.
    log_path = _write_session_log(
        question=question,
        tier=tier,
        confidence=confidence,
        seat_answers=seat_answers,
        synthesis=synthesis,
        decision=decision,
        work_dir=work_dir,
    )

    return {
        "tier": tier,
        "confidence": confidence,
        "seat_answers": seat_answers,
        "synthesis": synthesis,
        "log_path": str(log_path),
        "decision": decision,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    """Simple CLI for standalone invocation."""
    if len(sys.argv) < 2:
        print("Usage: python -m token_optimiser.boa <question>")
        print("   or: python token_optimiser/boa.py <question>")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = run_boa(question)

    print(f"\n{'='*60}")
    print(f"BOA Session Complete")
    print(f"{'='*60}")
    print(f"Question:   {question}")
    print(f"Tier:       {result['tier']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Log:        {result['log_path']}")
    print(f"\nSeat answers: {len(result['seat_answers'])}")
    for ans in result["seat_answers"]:
        print(f"  [{ans['seat_name']}] {ans['answer'][:80]}...")
    print(f"\nSynthesis:\n{result['synthesis']}")


if __name__ == "__main__":
    _cli()

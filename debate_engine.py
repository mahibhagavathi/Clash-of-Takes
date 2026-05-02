"""
Debate Arena — Debate Engine
Handles argument generation for individual agents.
Designed to be stateless: all debate history is passed in on each call,
making it compatible with Streamlit's rerun-based state management.
"""

import json
from groq import Groq
from personas import PERSONAS

# ── Constants ────────────────────────────────────────────────────────────────

ROUND_LABELS: list[str] = [
    "Opening Argument",
    "Rebuttal",
    "Counter-Rebuttal",
    "Deeper Rebuttal",
    "Closing Statement",
]

DEPTH_TO_ROUNDS: dict[str, int] = {
    "Short  (2 rounds)": 2,
    "Medium (3 rounds)": 3,
    "Deep   (5 rounds)": 5,
}

STYLE_INSTRUCTIONS: dict[str, str] = {
    "Formal":   "Be structured and strictly logical. Build from clear premises to conclusions. No emotional appeals.",
    "Heated":   "Be aggressive and passionate. Directly challenge your opponent. Show raw conviction and urgency.",
    "Academic": "Be analytical and evidence-based. Cite specific data, examples, and thinkers. Be precise and thorough.",
    "Casual":   "Be conversational and accessible. Use everyday language, analogies, and a natural speaking tone.",
}

MODEL = "llama-3.3-70b-versatile"

# ── Agent ─────────────────────────────────────────────────────────────────────

class DebateAgent:
    """A single AI debater bound to a specific ideological persona."""

    def __init__(self, persona_name: str, api_key: str) -> None:
        self.persona_name = persona_name
        self.persona = PERSONAS[persona_name]
        self.client = Groq(api_key=api_key)

    def generate_argument(
        self,
        topic: str,
        round_num: int,
        total_rounds: int,
        style: str,
        debate_history: list[dict],
        opponent_name: str,
    ) -> str:
        """
        Generate the agent's argument for the current round.

        Args:
            topic:          The debate topic.
            round_num:      Current round number (1-indexed).
            total_rounds:   Total number of rounds in this debate.
            style:          Debate style key (see STYLE_INSTRUCTIONS).
            debate_history: All entries generated so far.
            opponent_name:  The opposing persona's name.

        Returns:
            The agent's argument as a plain-text string.
        """
        round_label = ROUND_LABELS[min(round_num - 1, len(ROUND_LABELS) - 1)]
        style_instruction = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["Formal"])

        # Build readable history context
        history_block = ""
        if debate_history:
            history_block = "\n\n=== DEBATE SO FAR ===\n"
            for entry in debate_history:
                history_block += (
                    f"\n[{entry['persona']}] — {entry['round_label']}:\n"
                    f"{entry['text']}\n"
                )

        # Craft the instruction based on which round we're in
        if round_num == 1:
            instruction = (
                f'DEBATE TOPIC: "{topic}"\n'
                f"YOUR OPPONENT REPRESENTS: {opponent_name}\n"
                f"ROUND {round_num} OF {total_rounds}: {round_label}\n\n"
                f"STYLE DIRECTIVE: {style_instruction}\n\n"
                "Present your opening argument. Be specific, compelling, and stay fully in character. "
                "Aim for 3–4 paragraphs of flowing prose. Do not use bullet points."
            )
        elif round_num == total_rounds:
            instruction = (
                f'DEBATE TOPIC: "{topic}"\n'
                f"{history_block}\n\n"
                f"ROUND {round_num} OF {total_rounds}: {round_label} (FINAL)\n\n"
                f"STYLE DIRECTIVE: {style_instruction}\n\n"
                "Give your closing statement. Summarize your strongest points, expose the fatal flaws "
                "in your opponent's position, and end with a powerful conclusion. "
                "3–4 paragraphs. Stay in character. Do not repeat yourself verbatim."
            )
        else:
            instruction = (
                f'DEBATE TOPIC: "{topic}"\n'
                f"{history_block}\n\n"
                f"ROUND {round_num} OF {total_rounds}: {round_label}\n\n"
                f"STYLE DIRECTIVE: {style_instruction}\n\n"
                "Directly respond to your opponent's most recent argument. "
                "Attack their weakest points, defend your position, and advance new supporting evidence. "
                "3–4 paragraphs of flowing prose. Stay in character. Do not repeat prior arguments verbatim."
            )

        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": self.persona["system_prompt"]},
                {"role": "user",   "content": instruction},
            ],
            max_tokens=650,
            temperature=0.85,
        )
        return response.choices[0].message.content.strip()


# ── Post-debate utilities ─────────────────────────────────────────────────────

def generate_synthesis(
    topic: str,
    history: list[dict],
    persona_a: str,
    persona_b: str,
    api_key: str,
) -> str:
    """Generate an impartial synthesis of the completed debate."""
    client = Groq(api_key=api_key)

    history_text = ""
    for entry in history:
        history_text += f"\n[{entry['persona']}] ({entry['round_label']}):\n{entry['text']}\n\n"

    prompt = (
        f'You are an impartial debate analyst. Analyze this debate on: "{topic}"\n\n'
        f"{history_text}\n\n"
        "Provide a balanced synthesis using these five sections:\n\n"
        f"**Core Tension** — What is the fundamental disagreement between the two sides?\n\n"
        f"**Strongest Points — {persona_a}** — Their 2–3 most compelling arguments.\n\n"
        f"**Strongest Points — {persona_b}** — Their 2–3 most compelling arguments.\n\n"
        "**Key Weaknesses** — What did each side fail to address or adequately defend?\n\n"
        "**Takeaway** — Where might both sides find agreement, and where is the disagreement genuinely irresolvable?\n\n"
        "Be concise, fair, and intellectually honest. Write in clear prose."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


def get_ai_judgment(
    topic: str,
    history: list[dict],
    persona_a: str,
    persona_b: str,
    api_key: str,
) -> dict:
    """
    Ask an AI judge to score both debaters and declare a winner.
    Returns a dict with scores for both agents and a winner.
    """
    client = Groq(api_key=api_key)

    history_text = ""
    for entry in history:
        history_text += f"\n[{entry['persona']}] ({entry['round_label']}):\n{entry['text']}\n\n"

    prompt = (
        f'You are a strict, impartial debate judge. Evaluate this debate on: "{topic}"\n\n'
        f"{history_text}\n\n"
        "Score each debater from 1–10 on four criteria:\n"
        "- logic: Quality of logical structure and reasoning\n"
        "- evidence: Use of examples, data, and real-world cases\n"
        "- rebuttal: Effectiveness at dismantling the opponent's arguments\n"
        "- persuasion: Overall persuasive impact\n\n"
        "Then declare a winner. Respond ONLY with valid JSON — no markdown, no preamble:\n"
        "{\n"
        f'  "agent_a": {{"persona": "{persona_a}", "logic": 0, "evidence": 0, "rebuttal": 0, "persuasion": 0, "total": 0}},\n'
        f'  "agent_b": {{"persona": "{persona_b}", "logic": 0, "evidence": 0, "rebuttal": 0, "persuasion": 0, "total": 0}},\n'
        '  "winner": "<persona_name>",\n'
        '  "reason": "<one concise sentence explaining the decision>"\n'
        "}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    if "```" in raw:
        parts = raw.split("```")
        # Take the content inside the first fence pair
        raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

    result = json.loads(raw)

    # Auto-compute totals if the model didn't
    for key in ("agent_a", "agent_b"):
        a = result[key]
        if a.get("total", 0) == 0:
            a["total"] = a["logic"] + a["evidence"] + a["rebuttal"] + a["persuasion"]

    return result

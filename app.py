"""
Debate Arena — Main Streamlit App
Multi-agent AI debate system powered by Groq (llama-3.3-70b-versatile).

State machine:
  "idle"     → user is configuring the debate
  "running"  → debate is auto-advancing, one argument per page rerun
  "complete" → all rounds done, synthesis/judgment available
"""

import random
import time

import streamlit as st

from debate_engine import (
    DEPTH_TO_ROUNDS,
    ROUND_LABELS,
    STYLE_INSTRUCTIONS,
    DebateAgent,
    generate_synthesis,
    get_ai_judgment,
)
from personas import AUTO_PAIRS, PERSONAS

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Debate Arena ⚔️",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Tighten default padding */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    /* Round header pill */
    .round-header {
        background: linear-gradient(90deg, #1e1e2e, #2a2a3e);
        border-left: 4px solid #7c3aed;
        border-radius: 0 8px 8px 0;
        padding: 0.5rem 1rem;
        margin: 1.5rem 0 0.75rem 0;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #c4b5fd;
    }

    /* Persona card in sidebar */
    .persona-card {
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin: 0.25rem 0;
        font-size: 0.82rem;
        font-weight: 500;
        color: #ffffff;
    }

    /* Winner banner */
    .winner-banner {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        text-align: center;
        margin: 1rem 0;
        color: #1c1c1c;
        font-weight: 700;
        font-size: 1.3rem;
    }

    /* Score metric card */
    .score-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Running status bar */
    .status-running {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        color: #8b949e;
        font-size: 0.85rem;
        text-align: center;
        animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state init ────────────────────────────────────────────────────────

DEFAULTS = {
    "app_state":      "idle",     # "idle" | "running" | "complete"
    "debate_config":  {},
    "debate_history": [],
    "current_round":  1,
    "current_agent":  "A",        # "A" | "B"
    "synthesis":      "",
    "judgment":       {},
    "user_vote":      "",
    "error_msg":      "",
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Helper: read API key ──────────────────────────────────────────────────────

# ── Helper: display one debate message ───────────────────────────────────────

def display_message(entry: dict, show_round_header: bool = False) -> None:
    if show_round_header:
        st.markdown(
            f'<div class="round-header">⚔️ Round {entry["round_num"]} — {entry["round_label"]}</div>',
            unsafe_allow_html=True,
        )
    with st.chat_message(name=entry["persona"], avatar=entry["emoji"]):
        st.markdown(f"**{entry['persona']}**")
        st.write(entry["text"])

# ── Helper: display scorecard ─────────────────────────────────────────────────

def display_scorecard(judgment: dict) -> None:
    winner_name = judgment.get("winner", "")
    reason      = judgment.get("reason", "")

    st.markdown(
        f'<div class="winner-banner">🏆 Winner: {winner_name}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"*Judge's reasoning: {reason}*")
    st.write("")

    col_a, col_b = st.columns(2)
    for col, key in ((col_a, "agent_a"), (col_b, "agent_b")):
        data    = judgment[key]
        persona = data["persona"]
        color   = PERSONAS[persona]["color"]
        emoji   = PERSONAS[persona]["emoji"]
        is_win  = persona == winner_name

        with col:
            st.markdown(
                f'<div class="persona-card" style="background:{color};">'
                f'{emoji} {persona}{"  🏆" if is_win else ""}'
                f"</div>",
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            m1.metric("Logic",      f"{data['logic']}/10")
            m2.metric("Evidence",   f"{data['evidence']}/10")
            m3, m4 = st.columns(2)
            m3.metric("Rebuttal",   f"{data['rebuttal']}/10")
            m4.metric("Persuasion", f"{data['persuasion']}/10")
            st.markdown(f"**Total: {data['total']} / 40**")

# ── Helper: build an entry dict ───────────────────────────────────────────────

def make_entry(agent_key: str, persona_name: str, round_num: int, text: str) -> dict:
    p = PERSONAS[persona_name]
    label = ROUND_LABELS[min(round_num - 1, len(ROUND_LABELS) - 1)]
    return {
        "agent":       agent_key,
        "persona":     persona_name,
        "emoji":       p["emoji"],
        "color":       p["color"],
        "round_num":   round_num,
        "round_label": label,
        "text":        text,
    }

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚔️ Debate Arena")
    st.caption("Multi-agent AI debate system powered by Groq")
    st.divider()

    # API key (hidden if already in secrets)
    try:
        _ = st.secrets["GROQ_API_KEY"]
        st.success("✅ Groq API key loaded from secrets")
        sidebar_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        sidebar_api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get a free key at console.groq.com",
            key="sidebar_api_key",
        )
        if not sidebar_api_key:
            st.info("Enter your Groq API key to start.")

    st.divider()

    # Disable config inputs while debate is running
    locked = st.session_state.app_state in ("running", "complete")

    topic = st.text_area(
        "📝 Debate Topic",
        placeholder="e.g. Should AI replace teachers?",
        height=80,
        disabled=locked,
        key="sidebar_topic",
    )

    st.write("**🤺 Personas**")
    persona_options = ["Auto (best match)"] + list(PERSONAS.keys())

    col_a, col_b = st.columns(2)
    with col_a:
        persona_a_sel = st.selectbox(
            "Side A",
            persona_options,
            disabled=locked,
            key="sidebar_pa",
        )
    with col_b:
        persona_b_sel = st.selectbox(
            "Side B",
            persona_options,
            disabled=locked,
            key="sidebar_pb",
        )

    style_sel = st.selectbox(
        "🎭 Debate Style",
        list(STYLE_INSTRUCTIONS.keys()),
        disabled=locked,
        key="sidebar_style",
    )

    depth_sel = st.selectbox(
        "⏱ Depth",
        list(DEPTH_TO_ROUNDS.keys()),
        index=1,
        disabled=locked,
        key="sidebar_depth",
    )

    st.divider()

    # Show persona preview cards
    if st.session_state.app_state == "idle":
        # Resolve auto selection for preview
        def _resolve(sel: str, other: str | None = None) -> str:
            if sel != "Auto (best match)":
                return sel
            available = [k for k in PERSONAS if k != other]
            return random.choice(available)

        preview_a = _resolve(persona_a_sel)
        preview_b = _resolve(persona_b_sel, preview_a)

        for pname in (preview_a, preview_b):
            p = PERSONAS[pname]
            st.markdown(
                f'<div class="persona-card" style="background:{p["color"]};">'
                f'{p["emoji"]} <b>{pname}</b><br>'
                f'<span style="font-weight:400;opacity:0.85;">{p["tagline"]}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

    # Start / Reset buttons
    st.write("")
    if st.session_state.app_state == "idle":
        ready = bool(sidebar_api_key and topic and topic.strip())
        if st.button(
            "⚔️ Start Debate",
            use_container_width=True,
            type="primary",
            disabled=not ready,
        ):
            # Resolve personas
            def _pick(sel: str, exclude: str = "") -> str:
                if sel != "Auto (best match)":
                    return sel
                pairs = [p for p in AUTO_PAIRS if exclude not in p or True]
                good  = [p for p in AUTO_PAIRS]
                pa, pb = random.choice(good)
                return pa if not exclude else (pb if pa == exclude else pa)

            pa = _pick(persona_a_sel)
            pb = _pick(persona_b_sel, pa) if persona_b_sel == "Auto (best match)" else persona_b_sel
            if pa == pb:
                # Find a different persona
                pb = next(k for k in PERSONAS if k != pa)

            total = DEPTH_TO_ROUNDS[depth_sel]

            st.session_state.debate_config  = {
                "topic":        topic.strip(),
                "persona_a":    pa,
                "persona_b":    pb,
                "style":        style_sel,
                "depth":        depth_sel,
                "total_rounds": total,
            }
            st.session_state.debate_history = []
            st.session_state.current_round  = 1
            st.session_state.current_agent  = "A"
            st.session_state.synthesis      = ""
            st.session_state.judgment       = {}
            st.session_state.user_vote      = ""
            st.session_state.error_msg      = ""
            st.session_state.app_state      = "running"
            st.rerun()

    elif st.session_state.app_state in ("running", "complete"):
        if st.button("🔄 New Debate", use_container_width=True):
            for key, val in DEFAULTS.items():
                st.session_state[key] = val
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Landing screen ────────────────────────────────────────────────────────────

if st.session_state.app_state == "idle":
    st.title("⚔️ Debate Arena")
    st.markdown(
        "**A multi-agent AI system where ideologically distinct personas clash in structured debates.**  \n"
        "Configure your debate in the sidebar, then hit **Start Debate**."
    )
    st.write("")

    # Feature grid
    c1, c2, c3, c4 = st.columns(4)
    c1.info("🤖 **Multi-Agent**\nIndependent AI personas argue their worldview")
    c2.info("🧠 **Structured Rounds**\nOpening → Rebuttal → Counter → Close")
    c3.info("⚖️ **AI Judge**\nScored on logic, evidence, rebuttal & persuasion")
    c4.info("🗳️ **You Decide**\nVote for the winner yourself")

    st.write("")
    st.subheader("Available Personas")
    cols = st.columns(4)
    for i, (pname, pdata) in enumerate(PERSONAS.items()):
        with cols[i % 4]:
            st.markdown(
                f'<div class="persona-card" style="background:{pdata["color"]};margin-bottom:0.5rem;">'
                f'{pdata["emoji"]} <b>{pname}</b><br>'
                f'<span style="opacity:0.85;font-weight:400;">{pdata["tagline"]}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

# ── Debate view (running or complete) ─────────────────────────────────────────

else:
    cfg = st.session_state.debate_config
    pa, pb = cfg["persona_a"], cfg["persona_b"]
    total  = cfg["total_rounds"]

    # Header
    st.title("⚔️ Debate Arena")
    h_col1, h_col2, h_col3 = st.columns([2, 1, 1])
    with h_col1:
        st.markdown(f"### 📝 {cfg['topic']}")
    with h_col2:
        pa_data = PERSONAS[pa]
        st.markdown(
            f'<div class="persona-card" style="background:{pa_data["color"]};text-align:center;">'
            f'{pa_data["emoji"]} {pa}</div>',
            unsafe_allow_html=True,
        )
    with h_col3:
        pb_data = PERSONAS[pb]
        st.markdown(
            f'<div class="persona-card" style="background:{pb_data["color"]};text-align:center;">'
            f'{pb_data["emoji"]} {pb}</div>',
            unsafe_allow_html=True,
        )

    st.caption(f"Style: **{cfg['style']}** · Depth: **{cfg['depth']}** · {total} rounds")

    rounds_done = (
        st.session_state.current_round - 1
        if st.session_state.current_agent == "A"
        else st.session_state.current_round - 1
    )
    progress_val = len(st.session_state.debate_history) / (total * 2)
    st.progress(min(progress_val, 1.0))
    st.divider()

    # ── Display all history so far ────────────────────────────────────────────
    last_round_shown = 0
    for entry in st.session_state.debate_history:
        show_header = entry["round_num"] != last_round_shown
        display_message(entry, show_round_header=show_header)
        last_round_shown = entry["round_num"]

    # ── Error display ─────────────────────────────────────────────────────────
    if st.session_state.error_msg:
        st.error(st.session_state.error_msg)
        if st.button("🔁 Retry"):
            st.session_state.error_msg = ""
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # RUNNING STATE: generate the next argument then rerun
    # ══════════════════════════════════════════════════════════════════════════

    elif st.session_state.app_state == "running":
        round_num   = st.session_state.current_round
        agent_key   = st.session_state.current_agent
        persona_now = pa if agent_key == "A" else pb
        opponent    = pb if agent_key == "A" else pa

        # Check if we're done
        if round_num > total:
            st.session_state.app_state = "complete"
            st.rerun()

        # Show "thinking" indicator
        p = PERSONAS[persona_now]
        st.markdown(
            f'<div class="status-running">💭 {p["emoji"]} <b>{persona_now}</b> is formulating their argument…</div>',
            unsafe_allow_html=True,
        )

        # Generate argument
        try:
            agent = DebateAgent(persona_name=persona_now, api_key=sidebar_api_key)
            text  = agent.generate_argument(
                topic          = cfg["topic"],
                round_num      = round_num,
                total_rounds   = total,
                style          = cfg["style"],
                debate_history = st.session_state.debate_history,
                opponent_name  = opponent,
            )
        except Exception as exc:
            st.session_state.error_msg = f"API error: {exc}"
            st.rerun()
            st.stop()

        # Save and advance state
        entry = make_entry(agent_key, persona_now, round_num, text)
        st.session_state.debate_history.append(entry)

        if agent_key == "A":
            st.session_state.current_agent = "B"
        else:
            st.session_state.current_agent  = "A"
            st.session_state.current_round += 1

        time.sleep(0.2)   # brief pause so spinner is visible
        st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # COMPLETE STATE: synthesis + judgment
    # ══════════════════════════════════════════════════════════════════════════

    elif st.session_state.app_state == "complete":
        st.success("✅ Debate complete!")
        st.write("")

        # ── Synthesis ─────────────────────────────────────────────────────────
        st.subheader("📋 Synthesis")

        if not st.session_state.synthesis:
            with st.spinner("Generating synthesis…"):
                try:
                    st.session_state.synthesis = generate_synthesis(
                        topic     = cfg["topic"],
                        history   = st.session_state.debate_history,
                        persona_a = pa,
                        persona_b = pb,
                        api_key   = sidebar_api_key,
                    )
                except Exception as exc:
                    st.session_state.synthesis = f"⚠️ Could not generate synthesis: {exc}"

        st.markdown(st.session_state.synthesis)
        st.divider()

        # ── Judgment ──────────────────────────────────────────────────────────
        st.subheader("🏆 Who Won?")

        tab_ai, tab_vote = st.tabs(["🤖 AI Judge", "🗳️ Vote Yourself"])

        with tab_ai:
            if not st.session_state.judgment:
                if st.button("⚖️ Get AI Verdict", type="primary"):
                    with st.spinner("Judge is deliberating…"):
                        try:
                            st.session_state.judgment = get_ai_judgment(
                                topic     = cfg["topic"],
                                history   = st.session_state.debate_history,
                                persona_a = pa,
                                persona_b = pb,
                                api_key   = sidebar_api_key,
                            )
                        except Exception as exc:
                            st.error(f"Judgment failed: {exc}")
                    st.rerun()
            else:
                display_scorecard(st.session_state.judgment)

        with tab_vote:
            if not st.session_state.user_vote:
                st.write("Who made the stronger case?")
                v1, v2 = st.columns(2)

                pa_d = PERSONAS[pa]
                pb_d = PERSONAS[pb]

                with v1:
                    if st.button(
                        f"{pa_d['emoji']} {pa}",
                        use_container_width=True,
                        type="primary",
                    ):
                        st.session_state.user_vote = pa
                        st.rerun()
                with v2:
                    if st.button(
                        f"{pb_d['emoji']} {pb}",
                        use_container_width=True,
                        type="primary",
                    ):
                        st.session_state.user_vote = pb
                        st.rerun()
            else:
                winner = st.session_state.user_vote
                st.markdown(
                    f'<div class="winner-banner">🏆 You voted: {winner}</div>',
                    unsafe_allow_html=True,
                )
                st.balloons()
                st.write("")
                if st.button("Change vote"):
                    st.session_state.user_vote = ""
                    st.rerun()

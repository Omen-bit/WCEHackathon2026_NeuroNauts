"""
study_hub.py — Active Recall Study Hub
Provides adaptive AP/College-level quiz generation, interactive 3D active-recall flashcards,
and automated chapter study guide exports grounded in OpenStax Psychology 2e.
"""

import html
import json
import streamlit as st

CHAPTER_OPTIONS = [
    "Chapter 1: Introduction to Psychology",
    "Chapter 2: Psychological Research",
    "Chapter 3: Biopsychology & Neural Communication",
    "Chapter 4: States of Consciousness & Sleep",
    "Chapter 5: Sensation and Perception",
    "Chapter 6: Learning & Conditioning",
    "Chapter 7: Thinking and Intelligence",
    "Chapter 8: Memory Systems & Forgetting",
    "Chapter 9: Lifespan Development",
    "Chapter 10: Emotion and Motivation",
    "Chapter 11: Personality Theories",
    "Chapter 12: Social Psychology & Influence",
    "Chapter 14: Stress, Lifestyle, and Health",
    "Chapter 15: Psychological Disorders (DSM-5)",
    "Chapter 16: Therapy and Treatment"
]

DEFAULT_FLASHCARDS = {
    "Chapter 6: Learning & Conditioning": [
        {
            "id": 1,
            "term": "Classical Conditioning (Pavlovian)",
            "category": "Behavioral Learning",
            "definition": "A learning process where a biologically potent stimulus (Unconditioned Stimulus, UCS) is paired with a previously neutral stimulus (Conditioned Stimulus, CS) to elicit an involuntary conditioned response (CR)."
        },
        {
            "id": 2,
            "term": "Operant Conditioning (Skinner)",
            "category": "Behavioral Mechanism",
            "definition": "A form of learning where behaviors are strengthened or weakened through the delivery of reinforcers (increasing behavior frequency) or punishers (decreasing behavior frequency)."
        },
        {
            "id": 3,
            "term": "Positive vs. Negative Reinforcement",
            "category": "Core Principle",
            "definition": "Positive Reinforcement ADDS a desirable stimulus to increase behavior (e.g. praise). Negative Reinforcement REMOVES an aversive stimulus to increase behavior (e.g. turning off a loud alarm)."
        },
        {
            "id": 4,
            "term": "Extinction & Spontaneous Recovery",
            "category": "Conditioning Dynamics",
            "definition": "Extinction occurs when the conditioned stimulus is presented repeatedly without the UCS, weakening the CR. Spontaneous Recovery is the sudden reappearance of an extinguished CR after a rest period."
        },
        {
            "id": 5,
            "term": "Variable Ratio Schedule (VR)",
            "category": "Reinforcement Schedule",
            "definition": "Reinforcement is delivered after an unpredictable number of responses (e.g., slot machines). Produces the highest response rate and highest resistance to extinction."
        },
        {
            "id": 6,
            "term": "Observational Learning & Modeling",
            "category": "Bandura Social Learning",
            "definition": "Learning by observing the behavior of others and the consequences of their actions, famously demonstrated in Albert Bandura's Bobo Doll experiment."
        }
    ],
    "Chapter 8: Memory Systems & Forgetting": [
        {
            "id": 1,
            "term": "Atkinson-Shiffrin Model",
            "category": "Memory Architecture",
            "definition": "Memory passes through three distinct stages: Sensory Memory (fleeting raw input) → Short-Term Memory / Working Memory (retained ~20-30 sec) → Long-Term Memory (permanent storage)."
        },
        {
            "id": 2,
            "term": "Proactive vs. Retroactive Interference",
            "category": "Forgetting Mechanisms",
            "definition": "Proactive Interference: Old information hinders the recall of newly learned information. Retroactive Interference: Newly learned information impairs the retrieval of previously stored memories."
        },
        {
            "id": 3,
            "term": "Anterograde vs. Retrograde Amnesia",
            "category": "Clinical Pathology",
            "definition": "Anterograde Amnesia: Inability to form new long-term declarative memories after trauma (e.g. Patient H.M.). Retrograde Amnesia: Loss of memories formed prior to the trauma event."
        },
        {
            "id": 4,
            "term": "Hippocampus Memory Consolidation",
            "category": "Neuroanatomy",
            "definition": "The medial temporal lobe structure essential for encoding and consolidating explicit (declarative) memories into the neocortex, though not the final storage site for procedural memory."
        },
        {
            "id": 5,
            "term": "Serial Position Effect",
            "category": "Cognitive Phenomenon",
            "definition": "The tendency to recall the first items (Primacy Effect, due to LTM transfer) and last items (Recency Effect, due to active working memory) in a list better than middle items."
        },
        {
            "id": 6,
            "term": "Chunking & Mnemonic Encoding",
            "category": "Encoding Strategy",
            "definition": "Organizing individual pieces of information into meaningful units or associative patterns, expanding short-term memory capacity beyond Miller's standard 7 ± 2 items."
        }
    ],
    "Chapter 3: Biopsychology & Neural Communication": [
        {
            "id": 1,
            "term": "Action Potential (All-or-None)",
            "category": "Neural Electrophysiology",
            "definition": "An electrical impulse generated when the neuronal membrane depolarizes to its threshold (~ -55 mV), triggering rapid influx of Na+ ions followed by K+ efflux along the axon."
        },
        {
            "id": 2,
            "term": "Synaptic Neurotransmission",
            "category": "Chemical Communication",
            "definition": "Action potentials trigger vesicle fusion at the terminal button, releasing neurotransmitters across the synaptic cleft to bind with postsynaptic receptor sites."
        },
        {
            "id": 3,
            "term": "Amygdala & Limbic Circuitry",
            "category": "Brain Anatomy",
            "definition": "Subcortical limbic structure critical for processing emotional valence, fear conditioning, vigilance, and initiating the sympathetic 'fight-or-flight' stress response."
        },
        {
            "id": 4,
            "term": "Neuroplasticity",
            "category": "Neural Adaptation",
            "definition": "The brain's dynamic capacity to reorganize synaptic connections, form new neural pathways, and adapt structurally in response to learning, environmental enrichment, or injury."
        }
    ]
}

FLASHCARD_CSS = """
<style>
.fc-card-wrapper {
    margin: 12px 0;
    width: 100%;
}
.fc-card {
    background: #FFFFFF;
    border: 1.5px solid #E2E8F0;
    border-radius: 16px;
    padding: 22px 26px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.fc-card:hover {
    border-color: #C7D2FE;
    box-shadow: 0 8px 24px rgba(79,70,229,0.12);
    transform: translateY(-2px);
}
.fc-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}
.fc-badge-front {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 3px 10px;
    border-radius: 20px;
    background: #EEF2FF;
    color: #4F46E5;
    border: 1px solid #C7D2FE;
}
.fc-term-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #0F172A;
    margin: 0 0 14px;
    letter-spacing: -0.02em;
}
.fc-definition-box {
    background: #0B0F19;
    border: 1.5px solid #312E81;
    border-radius: 12px;
    padding: 16px 20px;
    color: #F8FAFC;
    font-size: 0.92rem;
    line-height: 1.65;
    margin-top: 10px;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
}
.fc-def-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #A5B4FC;
    margin-bottom: 6px;
}
</style>
"""

def _generate_quiz_questions(groq_client, model: str, chapter: str, difficulty: str) -> list:
    prompt = f"""
Generate 4 high-yield multiple-choice questions for '{chapter}' from OpenStax Psychology 2e.
Difficulty Level: {difficulty}

Requirements:
- Each question must test conceptual understanding or experiment application, not trivial trivia.
- Include 4 distinct options: "A", "B", "C", "D".
- Indicate the single correct option key.
- Provide a clear rationale explaining why the correct option is right AND why common misconceptions are wrong.

Output MUST be a JSON list of objects with the exact schema:
[
  {{
    "id": 1,
    "question": "Question text here...",
    "options": {{
      "A": "Option A text",
      "B": "Option B text",
      "C": "Option C text",
      "D": "Option D text"
    }},
    "correct_option": "A",
    "explanation": "Detailed explanation of the psychological concept..."
  }}
]
Do NOT output any other text or markdown fences outside the JSON list.
"""
    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a psychology quiz engine that outputs pure JSON lists."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1400,
            timeout=45,
        )
        raw = completion.choices[0].message.content.strip()
        if raw.startswith("```json"): raw = raw[7:]
        if raw.startswith("```"): raw = raw[3:]
        if raw.endswith("```"): raw = raw[:-3]
        return json.loads(raw.strip())
    except Exception as e:
        return [
            {
                "id": 1,
                "question": f"Which of the following is a primary focus in {chapter}?",
                "options": {
                    "A": "Analyzing empirical behavioral data and underlying mechanisms",
                    "B": "Relying purely on introspection without empirical validation",
                    "C": "Rejecting the scientific method in favor of anecdotal claims",
                    "D": "Exclusively studying non-human subjects in artificial settings"
                },
                "correct_option": "A",
                "explanation": "Modern scientific psychology emphasizes empirical testing, controlled observation, and evidence-based mechanism analysis."
            }
        ]


def _generate_flashcards(groq_client, model: str, chapter: str) -> list:
    prompt = f"""
Generate 6 core active-recall flashcards for '{chapter}' from OpenStax Psychology 2e.

Requirements:
- Front: Key Term or Landmark Experiment or Psychological Law.
- Back: Concise definition, exact mechanism, and a concrete real-world example.

Output MUST be a JSON list of objects with this schema:
[
  {{
    "id": 1,
    "term": "Term or Concept Name",
    "category": "Theory / Mechanism / Experiment",
    "definition": "Clear concise explanation and real-world significance..."
  }}
]
Do NOT output any markdown fences or extraneous text outside the JSON list.
"""
    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a psychology flashcard generator that outputs valid JSON lists."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
            timeout=40,
        )
        raw = completion.choices[0].message.content.strip()
        if raw.startswith("```json"): raw = raw[7:]
        if raw.startswith("```"): raw = raw[3:]
        if raw.endswith("```"): raw = raw[:-3]
        return json.loads(raw.strip())
    except Exception as e:
        return DEFAULT_FLASHCARDS.get(chapter, DEFAULT_FLASHCARDS["Chapter 6: Learning & Conditioning"])


def show_study_hub_page(get_groq_client_fn, get_groq_model_fn):
    st.markdown(FLASHCARD_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0.4rem 0 1.2rem;max-width:960px;margin:0 auto;">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:0.6rem;">
            <div style="width:52px;height:52px;border-radius:14px;flex-shrink:0;
                        background:linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%);color:white;
                        display:flex;align-items:center;justify-content:center;
                        box-shadow:0 8px 20px rgba(124,58,237,0.28);">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.8" stroke="currentColor" style="width:28px;height:28px;">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
                </svg>
            </div>
            <div>
                <div style="display:flex;align-items:center;gap:8px;">
                    <h2 style="margin:0;font-size:1.65rem;font-weight:800;color:#0F172A;letter-spacing:-0.02em;">Active Recall Study Hub</h2>
                    <span style="font-size:0.68rem;font-weight:700;padding:2px 8px;border-radius:6px;background:#F5F3FF;color:#7C3AED;border:1px solid #DDD6FE;">ACTIVE RECALL</span>
                </div>
                <p style="margin:4px 0 0;font-size:0.86rem;color:#64748B;font-weight:500;">
                    Interactive flashcard decks &nbsp;·&nbsp; Adaptive diagnostic quizzes &nbsp;·&nbsp; Instant distractor analysis
                </p>
            </div>
        </div>
        <div style="height:1px;background:#E2E8F0;margin-top:1.2rem;"></div>
    </div>
    """, unsafe_allow_html=True)

    # Global Chapter Selector
    c_sel1, c_sel2 = st.columns([3, 1])
    with c_sel1:
        selected_chapter = st.selectbox("Select Study Chapter:", CHAPTER_OPTIONS, index=5)
    with c_sel2:
        quiz_diff = st.selectbox("Quiz Rigor:", ["College Standard", "AP Advanced / Clinical"], index=0)

    hub_tab1, hub_tab2 = st.tabs(["🃏 Active Recall Flashcards", "📝 Adaptive Diagnostic Quiz"])

    # ─── TAB 1: ACTIVE RECALL FLASHCARDS ──────────────────────────────────────
    with hub_tab1:
        st.markdown(f"<p style='font-size:0.88rem;color:#475569;font-weight:500;'>Interactive high-yield concept cards for <strong>{selected_chapter}</strong>. Click 'Reveal Mechanism' to test your active recall.</p>", unsafe_allow_html=True)

        # Initialize flashcard deck with defaults or cached AI cards
        if "cards_data" not in st.session_state or st.session_state.get("_last_cards_ch") != selected_chapter:
            # Preload default cards if available, else standard template
            st.session_state.cards_data = DEFAULT_FLASHCARDS.get(
                selected_chapter,
                DEFAULT_FLASHCARDS["Chapter 6: Learning & Conditioning"]
            )
            st.session_state._last_cards_ch = selected_chapter
            st.session_state.card_revealed = {}
            st.session_state.card_mastery = {}

        if "card_revealed" not in st.session_state:
            st.session_state.card_revealed = {}
        if "card_mastery" not in st.session_state:
            st.session_state.card_mastery = {}

        # Top Control Bar
        c_top1, c_top2 = st.columns([3, 1])
        with c_top1:
            # Calculate mastery progress
            cards = st.session_state.cards_data or []
            mastered_cnt = sum(1 for cid in st.session_state.card_mastery if st.session_state.card_mastery.get(cid) == "mastered")
            pct_mastered = round((mastered_cnt / max(len(cards), 1)) * 100)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;margin:8px 0;">
                <span style="font-size:0.85rem;font-weight:700;color:#0F172A;">Mastery: {mastered_cnt}/{len(cards)} ({pct_mastered}%)</span>
                <div style="flex:1;background:#E2E8F0;border-radius:10px;height:8px;overflow:hidden;">
                    <div style="width:{pct_mastered}%;background:linear-gradient(90deg, #4F46E5, #10B981);height:100%;border-radius:10px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c_top2:
            if st.button("✨ Generate AI Deck", type="secondary", use_container_width=True):
                with st.spinner("Synthesizing textbook concepts with Groq AI…"):
                    client = get_groq_client_fn()
                    model = get_groq_model_fn()
                    st.session_state.cards_data = _generate_flashcards(client, model, selected_chapter)
                    st.session_state.card_revealed = {}
                    st.session_state.card_mastery = {}
                    st.rerun()

        # Display Cards in 2-Column Responsive Grid
        c1, c2 = st.columns(2)
        for idx, c in enumerate(cards):
            col = c1 if idx % 2 == 0 else c2
            cid = f"{selected_chapter}_{idx}"
            is_revealed = st.session_state.card_revealed.get(cid, False)
            mastery_state = st.session_state.card_mastery.get(cid, None)

            term_escaped = html.escape(c.get("term", "Psychology Concept"))
            cat_escaped = html.escape(c.get("category", "Mechanism / Theory"))
            def_escaped = html.escape(c.get("definition", ""))

            with col:
                st.markdown(f"""
                <div class="fc-card-wrapper">
                    <div class="fc-card">
                        <div class="fc-header">
                            <span class="fc-badge-front">{cat_escaped}</span>
                            <span style="font-size:0.72rem;font-weight:700;color:{'#15803D' if mastery_state=='mastered' else ('#B45309' if mastery_state=='review' else '#94A3B8')};">
                                {'✓ MASTERED' if mastery_state=='mastered' else ('🔄 IN REVIEW' if mastery_state=='review' else 'CARD #' + str(idx+1))}
                            </span>
                        </div>
                        <div class="fc-term-title">{term_escaped}</div>
                        {f'<div class="fc-definition-box"><div class="fc-def-label">🧠 Mechanism & Definition</div>{def_escaped}</div>' if is_revealed else '<div style="font-size:0.8rem;color:#94A3B8;padding:12px 0;font-style:italic;">Click "Reveal Mechanism" below to check your understanding.</div>'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Interactive Action Bar below each card
                b_c1, b_c2, b_c3 = st.columns([2, 1, 1])
                with b_c1:
                    btn_label = "Hide ✕" if is_revealed else "Reveal Mechanism 👁️"
                    if st.button(btn_label, key=f"rev_{cid}", use_container_width=True):
                        st.session_state.card_revealed[cid] = not is_revealed
                        st.rerun()
                with b_c2:
                    if st.button("⭐ Got It", key=f"mast_{cid}", use_container_width=True):
                        st.session_state.card_mastery[cid] = "mastered"
                        st.session_state.card_revealed[cid] = True
                        st.rerun()
                with b_c3:
                    if st.button("🔄 Review", key=f"revw_{cid}", use_container_width=True):
                        st.session_state.card_mastery[cid] = "review"
                        st.session_state.card_revealed[cid] = True
                        st.rerun()

    # ─── TAB 2: ADAPTIVE QUIZ ──────────────────────────────────────────────────
    with hub_tab2:
        st.markdown(f"<p style='font-size:0.88rem;color:#475569;font-weight:500;'>Test your conceptual mastery of <strong>{selected_chapter}</strong> with instant distractor analysis.</p>", unsafe_allow_html=True)

        if "quiz_data" not in st.session_state or st.session_state.get("_last_quiz_ch") != selected_chapter:
            st.session_state.quiz_data = None
            st.session_state.user_answers = {}
            st.session_state.quiz_submitted = False

        if st.button("✨ Generate New Quiz", type="primary"):
            with st.spinner("Generating conceptual quiz items from OpenStax textbook…"):
                client = get_groq_client_fn()
                model = get_groq_model_fn()
                st.session_state.quiz_data = _generate_quiz_questions(client, model, selected_chapter, quiz_diff)
                st.session_state._last_quiz_ch = selected_chapter
                st.session_state.user_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()

        quiz_items = st.session_state.quiz_data
        if quiz_items:
            for idx, item in enumerate(quiz_items, 1):
                st.markdown(f"""
                <div style="background:#FFFFFF;border:1.5px solid #E2E8F0;border-radius:14px;padding:20px 24px;margin:1rem 0;box-shadow:0 2px 8px rgba(0,0,0,0.02);">
                    <div style="font-size:0.75rem;font-weight:800;color:#6366F1;margin-bottom:6px;letter-spacing:0.05em;">QUESTION {idx} OF {len(quiz_items)}</div>
                    <div style="font-size:1rem;font-weight:700;color:#0F172A;margin-bottom:12px;line-height:1.5;">{html.escape(item['question'])}</div>
                </div>
                """, unsafe_allow_html=True)

                opts = [f"{k}: {v}" for k, v in item['options'].items()]
                prev_choice = st.session_state.user_answers.get(f"q_{idx}")
                selected_opt = st.radio(
                    f"Options for Q{idx}",
                    opts,
                    key=f"rad_{idx}",
                    label_visibility="collapsed",
                    index=opts.index(prev_choice) if prev_choice in opts else None,
                    disabled=st.session_state.quiz_submitted
                )
                if selected_opt:
                    st.session_state.user_answers[f"q_{idx}"] = selected_opt

                if st.session_state.quiz_submitted:
                    chosen_key = selected_opt.split(":")[0].strip() if selected_opt else ""
                    is_correct = (chosen_key == item['correct_option'])
                    badge_color = "#15803D" if is_correct else "#B91C1C"
                    badge_bg = "#F0FDF4" if is_correct else "#FEF2F2"
                    badge_border = "#BBF7D0" if is_correct else "#FECACA"

                    st.markdown(f"""
                    <div style="background:{badge_bg};border:1.5px solid {badge_border};border-radius:10px;padding:14px 18px;margin:8px 0 18px;">
                        <span style="font-weight:800;color:{badge_color};font-size:0.9rem;">
                            {'✓ Correct!' if is_correct else f'✗ Incorrect — Correct Answer: Option {item["correct_option"]}'}
                        </span>
                        <p style="margin:6px 0 0;font-size:0.86rem;color:#334155;line-height:1.55;">
                            {html.escape(item['explanation'])}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            if not st.session_state.quiz_submitted:
                if st.button("📊 Submit Quiz for Grading", type="primary"):
                    st.session_state.quiz_submitted = True
                    st.rerun()
            else:
                # Calculate score
                correct_total = 0
                for idx, item in enumerate(quiz_items, 1):
                    ans = st.session_state.user_answers.get(f"q_{idx}", "")
                    if ans.startswith(item["correct_option"]):
                        correct_total += 1
                pct = round((correct_total / len(quiz_items)) * 100)
                st.markdown(f"""
                <div style="text-align:center;background:#FFFFFF;border:2px solid #E2E8F0;border-radius:16px;padding:24px;margin-top:1.5rem;box-shadow:0 4px 16px rgba(0,0,0,0.03);">
                    <div style="font-size:0.75rem;font-weight:800;color:#64748B;letter-spacing:0.06em;">OVERALL PERFORMANCE</div>
                    <div style="font-size:2.4rem;font-weight:800;color:#4F46E5;font-family:'JetBrains Mono', monospace;margin:4px 0;">{correct_total}/{len(quiz_items)} ({pct}%)</div>
                    <p style="font-size:0.88rem;color:#475569;margin:0;font-weight:500;">
                        {'Outstanding mastery of chapter concepts!' if pct >= 75 else 'Good attempt! Review the explanations above to strengthen weak areas.'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Click **✨ Generate New Quiz** above to generate interactive multiple-choice questions.")

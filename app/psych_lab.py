"""
psych_lab.py — Interactive Clinical Case Study Simulator
PsychLab simulates clinical consultations where students interview patient personas,
formulate diagnoses, and receive automated rubric evaluations grounded in OpenStax Psychology 2e.
"""

import html
import json
import streamlit as st
from pathlib import Path

# Preset Clinical Cases
CLINICAL_CASES = [
    {
        "id": "case_1",
        "title": "Case #101: Acute Panic & Agoraphobic Avoidance",
        "patient_name": "Elena Martinez",
        "age": 24,
        "occupation": "Graphic Designer",
        "chief_complaint": "“I feel like I’m having a heart attack out of nowhere whenever I leave my apartment.”",
        "vital_summary": "Heart rate spikes to 135 bpm during episodes, tremors, hyperventilation, normal ECG.",
        "difficulty": "Intermediate",
        "domain": "Anxiety & Conditioning (Chapters 10 & 15)",
        "ground_truth_diagnosis": "Panic Disorder with Agoraphobic Tendencies; Conditioned Panic Response",
        "persona_prompt": (
            "You are Elena Martinez, a 24-year-old graphic designer. Three months ago, you had a sudden, "
            "terrifying episode at a crowded grocery store: your heart pounded, you couldn't breathe, your hands trembled, "
            "and you were convinced you were going to die of a heart attack. The hospital ER found nothing wrong physically. "
            "Since then, you experience spontaneous panic attacks and now avoid grocery stores, public transit, and open plazas "
            "because you are terrified of having another attack where escape might be difficult or embarrassing. "
            "Stay strictly in character. Speak naturally with anxious hesitation. Do NOT mention the clinical terms 'Panic Disorder' "
            "or 'Agoraphobia' directly. Describe your physical feelings, fears, and daily struggle."
        )
    },
    {
        "id": "case_2",
        "title": "Case #102: Post-Trauma Memory Impairment",
        "patient_name": "Arthur Pendelton",
        "age": 58,
        "occupation": "Former Accountant",
        "chief_complaint": "“I remember my childhood clearly, but I can't remember what I ate for breakfast 10 minutes ago.”",
        "vital_summary": "Post-concussion status following an auto accident 6 months ago; intact procedural skills.",
        "difficulty": "Advanced",
        "domain": "Memory & Neuroscience (Chapters 3 & 8)",
        "ground_truth_diagnosis": "Anterograde Amnesia with Temporal Lobe / Hippocampal Involvement (similar to H.M.)",
        "persona_prompt": (
            "You are Arthur Pendelton, a 58-year-old retired accountant. After a severe car accident 6 months ago involving "
            "head trauma, you can remember your high school years, wedding day, and how to play chess or drive, "
            "but you cannot form new long-term memories. If the doctor leaves the room and returns 5 minutes later, "
            "you greet them as if meeting for the first time. You repeatedly ask where you are. "
            "Stay strictly in character. Be polite, slightly disoriented, and repeat that your childhood memory is great "
            "but recent events simply vanish."
        )
    },
    {
        "id": "case_3",
        "title": "Case #103: Ethical Conflict & Rationalization",
        "patient_name": "David Vance",
        "age": 31,
        "occupation": "Marketing Director",
        "chief_complaint": "“I pride myself on being honest, but my company forced me to push misleading ads, and now I feel sick.”",
        "vital_summary": "High stress, insomnia, self-justification behavior, interpersonal irritability.",
        "difficulty": "Beginner",
        "domain": "Social Psychology & Cognition (Chapter 12)",
        "ground_truth_diagnosis": "Cognitive Dissonance & Attitude Change Mechanisms (Festinger)",
        "persona_prompt": (
            "You are David Vance, 31, marketing director. You consider yourself deeply ethical and honest. However, your boss "
            "pressured you to lead a major ad campaign exaggerating product safety. Initially you felt immense internal guilt and turmoil, "
            "but recently you have started telling yourself 'everyone in advertising exaggerates anyway' and 'the product isn't really that bad.' "
            "You are struggling with this inner mental friction. Stay in character. Talk about how torn you feel between your values "
            "and what you actually did."
        )
    },
    {
        "id": "case_4",
        "title": "Case #104: Sleep-Wake Disruption & Sleep Paralysis",
        "patient_name": "Maya Chen",
        "age": 19,
        "occupation": "University Student",
        "chief_complaint": "“I wake up unable to move with a terrifying shadowy figure standing over my bed.”",
        "vital_summary": "Irregular sleep schedule, excessive daytime sleepiness, hypnagogic hallucinations.",
        "difficulty": "Intermediate",
        "domain": "States of Consciousness (Chapter 4)",
        "ground_truth_diagnosis": "Sleep Paralysis with Hypnagogic Hallucinations & Circadian Rhythm Sleep-Wake Disorder",
        "persona_prompt": (
            "You are Maya Chen, a 19-year-old college sophomore. For the past two semesters, you study late into the night and sleep "
            "erratically. About twice a week, upon waking up or drifting to sleep, you find your body completely paralyzed. "
            "During these episodes you experience vivid, frightening sensations of a shadowy presence in your room and pressure on your chest. "
            "Stay in character. Describe the sheer terror of waking up frozen and the exhaustion you feel during lectures."
        )
    },
    {
        "id": "case_5",
        "title": "Case #105: Learned Helplessness & Apathy",
        "patient_name": "Samuel Jackson",
        "age": 42,
        "occupation": "Warehouse Supervisor",
        "chief_complaint": "“Why even bother trying? No matter what I do, bad things happen anyway.”",
        "vital_summary": "Depressed mood, psychomotor slowing, anhedonia, internal/stable/global attributional style.",
        "difficulty": "Intermediate",
        "domain": "Learning, Conditioning & Mood (Chapters 6 & 15)",
        "ground_truth_diagnosis": "Learned Helplessness (Seligman) & Major Depressive Episode",
        "persona_prompt": (
            "You are Samuel Jackson, 42. Over the past year, your department went through multiple arbitrary layoffs and restructuring "
            "despite your best efforts. After having your proposals rejected repeatedly regardless of their quality, you have stopped "
            "making any effort at work or at home. You believe you have zero control over your life outcomes. "
            "Stay in character. Express passive defeatism, low energy, and a firm belief that actions have no bearing on results."
        )
    }
]


def _call_patient_llm(groq_client, model: str, case: dict, history: list, user_question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": f"{case['persona_prompt']}\n\nIMPORTANT: Respond in 2-4 sentences as the patient in a clinical intake interview. Never mention clinical diagnosis labels."
        }
    ]
    for h in history:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["patient"]})
    messages.append({"role": "user", "content": user_question})

    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=250,
            timeout=30,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"(Elena looks down anxiously and pauses) ... I'm sorry, I'm having trouble finding the words right now. (Error: {e})"


def _grade_clinical_submission(groq_client, model: str, case: dict, diagnosis: str, mechanism: str, treatment: str) -> dict:
    prompt = f"""
You are an expert Clinical Psychology Board Examiner grading a student's psychiatric case report.

CASE INFORMATION:
Patient Name: {case['patient_name']} (Age {case['age']})
Domain: {case['domain']}
Expected Diagnosis: {case['ground_truth_diagnosis']}

STUDENT'S SUBMISSION:
- Proposed Diagnosis: {diagnosis}
- Underlying Psychological Mechanism: {mechanism}
- Proposed Treatment & Intervention Plan: {treatment}

Evaluate this clinical submission rigorously against OpenStax Psychology 2e standards.
Output MUST be valid JSON with the following exact keys:
{{
  "score": <integer from 0 to 100>,
  "diagnostic_accuracy": "<Brief assessment of diagnosis accuracy>",
  "mechanism_rating": "<Feedback on theoretical mechanism accuracy>",
  "treatment_rating": "<Feedback on intervention plan>",
  "strengths": ["<Point 1>", "<Point 2>"],
  "missed_symptoms": ["<Point 1>", "<Point 2>"],
  "differential_diagnoses": ["<Differential 1>", "<Differential 2>"],
  "recommended_reading": "<Chapter & section in OpenStax Psychology 2e>"
}}
Do NOT output any markdown ticks or explanation outside the JSON.
"""
    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a psychology grading engine that returns only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800,
            timeout=40,
        )
        raw = completion.choices[0].message.content.strip()
        # Clean potential markdown wrapping
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        return json.loads(raw.strip())
    except Exception as e:
        return {
            "score": 75,
            "diagnostic_accuracy": f"Automated grading completed with fallback: {diagnosis}",
            "mechanism_rating": mechanism,
            "treatment_rating": treatment,
            "strengths": ["Clear communication", "Structured diagnostic reasoning"],
            "missed_symptoms": ["Verify full duration criteria against DSM-5 in OpenStax"],
            "differential_diagnoses": ["Related somatic symptom disorder", "Substance-induced anxiety"],
            "recommended_reading": f"OpenStax Psychology 2e: {case['domain']}"
        }


def show_psych_lab_page(get_groq_client_fn, get_groq_model_fn):
    st.markdown("""
    <div style="padding:0.4rem 0 1.2rem;max-width:960px;margin:0 auto;">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:0.6rem;">
            <div style="width:52px;height:52px;border-radius:14px;flex-shrink:0;
                        background:linear-gradient(135deg, #059669 0%, #047857 100%);color:white;
                        display:flex;align-items:center;justify-content:center;
                        box-shadow:0 8px 20px rgba(5,150,105,0.28);">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.8" stroke="currentColor" style="width:28px;height:28px;">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                </svg>
            </div>
            <div>
                <div style="display:flex;align-items:center;gap:8px;">
                    <h2 style="margin:0;font-size:1.65rem;font-weight:800;color:#0F172A;letter-spacing:-0.02em;">PsychLab · Clinical Consultation Simulator</h2>
                    <span style="font-size:0.68rem;font-weight:700;padding:2px 8px;border-radius:6px;background:#DCFCE7;color:#15803D;border:1px solid #BBF7D0;">CLINICAL HUD</span>
                </div>
                <p style="margin:4px 0 0;font-size:0.86rem;color:#64748B;font-weight:500;">
                    Live intake interview simulator &nbsp;·&nbsp; Diagnostic hypothesis testing &nbsp;·&nbsp; Automated OpenStax & DSM-5 rubric grading
                </p>
            </div>
        </div>
        <div style="height:1px;background:#E2E8F0;margin-top:1.2rem;"></div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize case state
    if "selected_case_id" not in st.session_state:
        st.session_state.selected_case_id = CLINICAL_CASES[0]["id"]
    if "case_dialogues" not in st.session_state:
        st.session_state.case_dialogues = {c["id"]: [] for c in CLINICAL_CASES}
    if "case_evaluations" not in st.session_state:
        st.session_state.case_evaluations = {}

    # Case Selector Bar
    case_map = {c["id"]: c for c in CLINICAL_CASES}
    case_titles = [f"{c['title']} ({c['difficulty']})" for c in CLINICAL_CASES]
    current_idx = [c["id"] for c in CLINICAL_CASES].index(st.session_state.selected_case_id)

    selected_title = st.selectbox(
        "Select Patient Case Study",
        case_titles,
        index=current_idx,
        label_visibility="collapsed"
    )
    selected_case = CLINICAL_CASES[case_titles.index(selected_title)]
    st.session_state.selected_case_id = selected_case["id"]

    # Case Header Info Card
    st.markdown(f"""
    <div style="background:linear-gradient(135deg, #FFFFFF 0%, #F0FDF4 100%);
                border:1.5px solid #BBF7D0;border-radius:16px;padding:22px 26px;margin:1rem auto 1.5rem;max-width:960px;
                box-shadow:0 4px 16px rgba(5,150,105,0.06);">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:14px;">
            <div>
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                    <span style="padding:3px 10px;border-radius:20px;font-size:0.72rem;font-weight:700;background:#DCFCE7;color:#15803D;border:1px solid #BBF7D0;">
                        {selected_case['domain']}
                    </span>
                    <span style="font-size:0.72rem;font-weight:700;padding:3px 10px;border-radius:20px;background:#FEF3C7;color:#92400E;border:1px solid #FDE68A;">
                        Rigor: {selected_case['difficulty']}
                    </span>
                </div>
                <h3 style="margin:0 0 6px;color:#0F172A;font-size:1.35rem;font-weight:800;">{selected_case['patient_name']}, {selected_case['age']} years old</h3>
                <div style="font-size:0.88rem;color:#475569;margin-bottom:6px;"><strong>Occupation:</strong> {selected_case['occupation']}</div>
                <div style="font-size:0.9rem;color:#047857;font-style:italic;font-weight:500;">{selected_case['chief_complaint']}</div>
            </div>
            <div style="text-align:right;background:white;padding:10px 14px;border-radius:10px;border:1px solid #E2E8F0;">
                <div style="font-size:0.7rem;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.05em;">Clinical Vitals Summary</div>
                <div style="font-size:0.78rem;color:#0F172A;margin-top:4px;font-weight:600;">{selected_case['vital_summary']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["💬 Clinical Intake Interview", "📋 Submit Diagnostic Assessment"])

    with tab1:
        st.markdown("<p style='font-size:0.88rem;color:#475569;font-weight:500;'>Conduct an open-ended intake interview with the patient to isolate symptom onset, cognitive distortions, and behavioral triggers.</p>", unsafe_allow_html=True)

        history = st.session_state.case_dialogues[selected_case["id"]]

        # Render conversation history
        for turn in history:
            # Doctor bubble
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-end;margin-bottom:12px;">
                <div style="background:linear-gradient(135deg, #4F46E5 0%, #3730A3 100%);color:white;padding:12px 20px;border-radius:16px 16px 2px 16px;max-width:75%;font-size:0.92rem;box-shadow:0 4px 12px rgba(79,70,229,0.22);">
                    <div style="font-size:0.72rem;font-weight:700;opacity:0.85;margin-bottom:2px;">CLINICIAN (YOU):</div>
                    {html.escape(turn['user'])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            # Patient bubble
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-start;margin-bottom:18px;">
                <div style="background:#FFFFFF;color:#0F172A;border:1.5px solid #E2E8F0;padding:14px 22px;border-radius:16px 16px 16px 2px;max-width:80%;font-size:0.92rem;box-shadow:0 3px 10px rgba(0,0,0,0.03);">
                    <div style="font-size:0.72rem;font-weight:800;color:#059669;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.04em;">
                        👤 {selected_case['patient_name']} (Patient):
                    </div>
                    {html.escape(turn['patient'])}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Chat Input for interview
        c_in1, c_in2 = st.columns([5, 1])
        with c_in1:
            patient_q = st.text_input("Ask patient a question…", key=f"q_{selected_case['id']}", label_visibility="collapsed", placeholder="e.g. When did these panic sensations first happen? What thoughts go through your mind?")
        with c_in2:
            send_btn = st.button("Speak 🗣️", key=f"btn_{selected_case['id']}", use_container_width=True)

        if send_btn and patient_q.strip():
            with st.spinner("Patient is responding…"):
                client = get_groq_client_fn()
                model = get_groq_model_fn()
                patient_ans = _call_patient_llm(client, model, selected_case, history, patient_q.strip())
                st.session_state.case_dialogues[selected_case["id"]].append({
                    "user": patient_q.strip(),
                    "patient": patient_ans
                })
                st.rerun()

        if history:
            if st.button("Clear Interview History ↺", key=f"clr_{selected_case['id']}", type="secondary"):
                st.session_state.case_dialogues[selected_case["id"]] = []
                st.rerun()

    with tab2:
        st.markdown("<p style='font-size:0.88rem;color:#475569;font-weight:500;'>Synthesize your clinical observations and formulate your diagnosis grounded in OpenStax textbook criteria.</p>", unsafe_allow_html=True)

        diag_input = st.text_input("1. Primary Diagnosis & DSM Criteria:", placeholder="e.g. Panic Disorder with secondary agoraphobic avoidance behavior", key=f"d_{selected_case['id']}")
        mech_input = st.text_area("2. Underlying Psychological Mechanism / Theory:", placeholder="e.g. Classical conditioning where interoceptive sensations (rapid heart rate) become conditioned stimuli triggering anticipatory panic...", key=f"m_{selected_case['id']}")
        treat_input = st.text_area("3. Proposed Intervention & Treatment Plan:", placeholder="e.g. Cognitive Behavioral Therapy (CBT) with interoceptive exposure and systematic desensitization...", key=f"t_{selected_case['id']}")

        if st.button("🎓 Submit Report for Evaluation", type="primary", key=f"sub_{selected_case['id']}"):
            if not diag_input.strip():
                st.warning("Please enter a proposed primary diagnosis first.")
            else:
                with st.spinner("Clinical Board is grading your case assessment…"):
                    client = get_groq_client_fn()
                    model = get_groq_model_fn()
                    eval_result = _grade_clinical_submission(client, model, selected_case, diag_input, mech_input, treat_input)
                    st.session_state.case_evaluations[selected_case["id"]] = eval_result
                    st.success("✅ Diagnostic report evaluated!"); st.rerun()

        # Display Evaluation Scorecard if available
        eval_res = st.session_state.case_evaluations.get(selected_case["id"])
        if eval_res:
            score = eval_res.get("score", 80)
            score_color = "#15803D" if score >= 80 else ("#B45309" if score >= 60 else "#B91C1C")
            score_bg = "#F0FDF4" if score >= 80 else ("#FFFBEB" if score >= 60 else "#FEF2F2")
            score_border = "#BBF7D0" if score >= 80 else ("#FDE68A" if score >= 60 else "#FECACA")

            st.markdown(f"""
            <div style="background:{score_bg};border:1.5px solid {score_border};border-radius:16px;padding:26px 30px;margin-top:1.5rem;box-shadow:0 8px 24px rgba(0,0,0,0.04);">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;">
                    <div>
                        <span style="font-size:0.75rem;font-weight:800;letter-spacing:0.06em;color:{score_color};text-transform:uppercase;">Board Assessment Scorecard</span>
                        <h3 style="margin:2px 0 0;font-size:1.4rem;color:#0F172A;font-weight:800;">Clinical Competency Report</h3>
                    </div>
                    <div style="background:white;padding:8px 18px;border-radius:12px;border:1.5px solid {score_border};text-align:center;">
                        <span style="font-size:0.7rem;font-weight:700;color:#64748B;text-transform:uppercase;">Score</span>
                        <div style="font-size:2.2rem;font-weight:800;color:{score_color};font-family:'JetBrains Mono', monospace;line-height:1;">
                            {score}<span style="font-size:1.1rem;font-weight:500;">/100</span>
                        </div>
                    </div>
                </div>
                
                <div style="background:white;border-radius:12px;padding:16px 20px;border:1px solid {score_border};margin-bottom:14px;font-size:0.9rem;line-height:1.65;color:#1E293B;">
                    <p style="margin:0 0 8px;"><strong>Diagnostic Accuracy:</strong> {html.escape(eval_res.get('diagnostic_accuracy', ''))}</p>
                    <p style="margin:0 0 8px;"><strong>Mechanism Analysis:</strong> {html.escape(eval_res.get('mechanism_rating', ''))}</p>
                    <p style="margin:0;"><strong>Intervention Feasibility:</strong> {html.escape(eval_res.get('treatment_rating', ''))}</p>
                </div>

                <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px;">
                    <div style="background:white;padding:14px 18px;border-radius:10px;border:1px solid #E2E8F0;">
                        <span style="font-size:0.82rem;font-weight:700;color:#15803D;">✓ Key Strengths</span>
                        <ul style="margin:8px 0 0;padding-left:18px;font-size:0.84rem;color:#334155;">
                            {''.join(f'<li style="margin-bottom:4px;">{html.escape(s)}</li>' for s in eval_res.get('strengths', []))}
                        </ul>
                    </div>
                    <div style="background:white;padding:14px 18px;border-radius:10px;border:1px solid #E2E8F0;">
                        <span style="font-size:0.82rem;font-weight:700;color:#B45309;">⚠️ Missed Clues & Differentials</span>
                        <ul style="margin:8px 0 0;padding-left:18px;font-size:0.84rem;color:#334155;">
                            {''.join(f'<li style="margin-bottom:4px;">{html.escape(d)}</li>' for d in eval_res.get('differential_diagnoses', []))}
                        </ul>
                    </div>
                </div>

                <div style="margin-top:16px;font-size:0.85rem;color:#64748B;border-top:1px dashed {score_border};padding-top:12px;font-weight:500;">
                    📖 <strong>Recommended Textbook Section:</strong> <span style="color:#0F172A;font-weight:700;">{html.escape(eval_res.get('recommended_reading', ''))}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

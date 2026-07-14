import streamlit as st
from orchestrator import run_research, run_url_research, run_pdf_research, run_comparison_research
from tools.pdf_export import generate_pdf
from tools.word_export import generate_word

st.set_page_config(
    page_title="Distill",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        max-width: 860px;
        padding: 3rem 2rem;
        margin: 0 auto;
    }
    @media (max-width: 768px) {
        .block-container { padding: 1.5rem 1rem; }
        .hero-title { font-size: 1.8rem !important; }
    }
    .wordmark {
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.15em;
        color: #888;
        text-transform: uppercase;
        margin-bottom: 2.5rem;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 500;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }
    .hero-sub {
        font-size: 15px;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .followup-label {
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .quick-answer {
        background: rgba(128,128,128,0.05);
        border-left: 2px solid rgba(128,128,128,0.3);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 14px;
        line-height: 1.7;
        margin-bottom: 8px;
    }
    .report-footer {
        font-size: 12px;
        color: #aaa;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 0.5px solid rgba(128,128,128,0.15);
    }
    .mode-label {
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.75rem;
    }
    .export-label {
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .stTextInput > div > div > input {
        border-radius: 8px !important;
        border: 0.5px solid rgba(128,128,128,0.3) !important;
        font-size: 15px !important;
        padding: 0.6rem 1rem !important;
        background: transparent !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: rgba(128,128,128,0.6) !important;
        box-shadow: none !important;
    }
    .stFormSubmitButton > button {
        border-radius: 8px !important;
        background: #1a1a1a !important;
        color: white !important;
        border: none !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        height: 44px !important;
        width: 100% !important;
    }
    .stFormSubmitButton > button:hover {
        opacity: 0.85 !important;
        background: #1a1a1a !important;
        color: white !important;
    }
    .stButton > button {
        border-radius: 8px !important;
        border: 0.5px solid rgba(128,128,128,0.2) !important;
        background: transparent !important;
        font-size: 13px !important;
        text-align: left !important;
        width: 100% !important;
        color: inherit !important;
    }
    .stButton > button:hover {
        border-color: rgba(128,128,128,0.5) !important;
        background: transparent !important;
    }
    .stDownloadButton > button {
        border-radius: 8px !important;
        background: transparent !important;
        color: inherit !important;
        border: 0.5px solid rgba(128,128,128,0.3) !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        width: 100% !important;
    }
    .stDownloadButton > button:hover {
        border-color: rgba(128,128,128,0.6) !important;
    }
    div[data-testid="stSelectbox"] > div {
        border-radius: 8px !important;
        border: 0.5px solid rgba(128,128,128,0.3) !important;
        font-size: 13px !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# session state
if "history" not in st.session_state:
    st.session_state.history = []
if "current_report" not in st.session_state:
    st.session_state.current_report = None
if "prefill_topic" not in st.session_state:
    st.session_state.prefill_topic = ""
if "research_mode" not in st.session_state:
    st.session_state.research_mode = "topic"
if "depth" not in st.session_state:
    st.session_state.depth = "normal"
if "tone" not in st.session_state:
    st.session_state.tone = "default"
if "searches_used" not in st.session_state:
    st.session_state.searches_used = 0

MAX_SEARCHES = 5

def is_valid_topic(topic: str):
    if len(topic.strip()) < 3:
        return False, "Topic too short — try something more specific."
    if len(topic.strip()) > 200:
        return False, "Topic too long — keep it under 200 characters."
    if topic.strip().startswith("http"):
        return False, "Looks like a URL — switch to URL mode."
    return True, ""

def store_result(result: dict):
    st.session_state.current_report = result
    st.session_state.searches_used += 1
    topics_in_history = [h["topic"] for h in st.session_state.history]
    if result["topic"] not in topics_in_history:
        st.session_state.history.insert(0, {
            "topic": result["topic"],
            "sources": result["sources_found"],
            "report": result["report"],
            "followups": result.get("followups", [])
        })
        st.session_state.history = st.session_state.history[:10]

def check_rate_limit() -> bool:
    if st.session_state.searches_used >= MAX_SEARCHES:
        st.warning(f"You've used all {MAX_SEARCHES} researches for this session. Refresh the page to start a new session.")
        return False
    return True

def run_and_store(topic: str, depth: str, tone: str):
    if not check_rate_limit():
        return
    valid, message = is_valid_topic(topic)
    if not valid:
        st.warning(message)
        return
    with st.status("Working...", expanded=True) as status:
        st.write("Searching the web...")
        try:
            result = run_research(topic, depth=depth, tone=tone)
            st.write(f"Reading {result['sources_found']} sources...")
            st.write("Writing your report...")
            status.update(label="Done", state="complete", expanded=False)
            store_result(result)
        except Exception as e:
            status.update(label="Something went wrong", state="error")
            st.error(f"{str(e)}")

def run_url_and_store(url: str, tone: str):
    if not check_rate_limit():
        return
    with st.status("Working...", expanded=True) as status:
        st.write("Reading the article...")
        try:
            result = run_url_research(url, tone=tone)
            st.write("Finding related sources...")
            st.write("Writing your report...")
            status.update(label="Done", state="complete", expanded=False)
            store_result(result)
        except Exception as e:
            status.update(label="Something went wrong", state="error")
            st.error(f"{str(e)}")

def run_pdf_and_store(uploaded_file, tone: str):
    if not check_rate_limit():
        return
    with st.status("Working...", expanded=True) as status:
        st.write("Reading your PDF...")
        try:
            result = run_pdf_research(uploaded_file, tone=tone)
            st.write("Finding related sources...")
            st.write("Writing your report...")
            status.update(label="Done", state="complete", expanded=False)
            store_result(result)
        except Exception as e:
            status.update(label="Something went wrong", state="error")
            st.error(f"{str(e)}")

def run_comparison_and_store(topic_a: str, topic_b: str, depth: str):
    if not check_rate_limit():
        return
    with st.status("Working...", expanded=True) as status:
        st.write(f"Researching {topic_a}...")
        try:
            result = run_comparison_research(topic_a, topic_b, depth=depth)
            st.write(f"Researching {topic_b}...")
            st.write("Writing comparison report...")
            status.update(label="Done", state="complete", expanded=False)
            store_result(result)
        except Exception as e:
            status.update(label="Something went wrong", state="error")
            st.error(f"{str(e)}")

# sidebar
with st.sidebar:
    st.markdown(
        '<div style="font-size:11px;font-weight:500;letter-spacing:0.1em;text-transform:uppercase;color:#888;margin-bottom:0.5rem">Recent Research</div>',
        unsafe_allow_html=True
    )
    remaining = MAX_SEARCHES - st.session_state.searches_used
    st.markdown(
        f'<div style="font-size:12px;color:#aaa;margin-bottom:1.5rem">{remaining} research{"es" if remaining != 1 else ""} remaining this session</div>',
        unsafe_allow_html=True
    )
    if not st.session_state.history:
        st.markdown(
            '<div style="font-size:13px;color:#aaa;margin-bottom:1rem">Your research history will appear here.</div>',
            unsafe_allow_html=True
        )
    else:
        for item in st.session_state.history:
            if st.button(f"↗ {item['topic']}", key=f"hist_{item['topic']}"):
                st.session_state.current_report = {
                    "topic": item["topic"],
                    "sources_found": item["sources"],
                    "report": item["report"],
                    "followups": item.get("followups", [])
                }
                st.rerun()

# main
st.markdown('<div class="wordmark">Distill</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-title">What do you want<br>to understand?</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="hero-sub">Drop a topic, URL, PDF, or compare two things. Get a research report distilled from the live web.</div>',
    unsafe_allow_html=True
)

# mode toggle
st.markdown('<div class="mode-label">Research by</div>', unsafe_allow_html=True)
col_t, col_u, col_p, col_c, col_empty = st.columns([1, 1, 1, 1, 2])
with col_t:
    if st.button(
        "✓ Topic" if st.session_state.research_mode == "topic" else "Topic",
        key="mode_topic", use_container_width=True
    ):
        st.session_state.research_mode = "topic"
        st.rerun()
with col_u:
    if st.button(
        "✓ URL" if st.session_state.research_mode == "url" else "URL",
        key="mode_url", use_container_width=True
    ):
        st.session_state.research_mode = "url"
        st.rerun()
with col_p:
    if st.button(
        "✓ PDF" if st.session_state.research_mode == "pdf" else "PDF",
        key="mode_pdf", use_container_width=True
    ):
        st.session_state.research_mode = "pdf"
        st.rerun()
with col_c:
    if st.button(
        "✓ Compare" if st.session_state.research_mode == "compare" else "Compare",
        key="mode_compare", use_container_width=True
    ):
        st.session_state.research_mode = "compare"
        st.rerun()

st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)

# topic mode
if st.session_state.research_mode == "topic":
    opt1, opt2, opt_empty = st.columns([1, 1, 4])
    with opt1:
        depth = st.selectbox(
            "Depth",
            options=["quick", "normal"],
            index=0 if st.session_state.depth == "quick" else 1,
            key="depth_select"
        )
        st.session_state.depth = depth
    with opt2:
        tone = st.selectbox(
            "Tone",
            options=["default", "academic", "executive", "beginner", "journalist"],
            index=["default","academic","executive","beginner","journalist"].index(st.session_state.tone),
            key="tone_select"
        )
        st.session_state.tone = tone

    with st.form(key="topic_form"):
        col1, col2 = st.columns([5, 1])
        with col1:
            topic = st.text_input(
                label="topic",
                placeholder="e.g. State of AI agents in 2026",
                label_visibility="collapsed",
                value=st.session_state.prefill_topic
            )
        with col2:
            submitted = st.form_submit_button("Distill →", use_container_width=True)
    if submitted and topic.strip():
        st.session_state.prefill_topic = ""
        run_and_store(topic.strip(), st.session_state.depth, st.session_state.tone)
    elif submitted:
        st.warning("Enter a topic to distill.")

# url mode
elif st.session_state.research_mode == "url":
    opt1, opt_empty = st.columns([1, 5])
    with opt1:
        tone = st.selectbox(
            "Tone",
            options=["default", "academic", "executive", "beginner", "journalist"],
            index=["default","academic","executive","beginner","journalist"].index(st.session_state.tone),
            key="url_tone_select"
        )
        st.session_state.tone = tone
    with st.form(key="url_form"):
        col1, col2 = st.columns([5, 1])
        with col1:
            url = st.text_input(
                label="url",
                placeholder="e.g. https://techcrunch.com/some-article",
                label_visibility="collapsed"
            )
        with col2:
            submitted = st.form_submit_button("Distill →", use_container_width=True)
    if submitted and url.strip():
        if not url.strip().startswith("http"):
            st.warning("Please enter a valid URL starting with http or https.")
        else:
            run_url_and_store(url.strip(), st.session_state.tone)
    elif submitted:
        st.warning("Enter a URL to distill.")

# pdf mode
elif st.session_state.research_mode == "pdf":
    opt1, opt_empty = st.columns([1, 5])
    with opt1:
        tone = st.selectbox(
            "Tone",
            options=["default", "academic", "executive", "beginner", "journalist"],
            index=["default","academic","executive","beginner","journalist"].index(st.session_state.tone),
            key="pdf_tone_select"
        )
        st.session_state.tone = tone
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        if st.button("Distill →", key="pdf_submit"):
            run_pdf_and_store(uploaded_file, st.session_state.tone)

# compare mode
elif st.session_state.research_mode == "compare":
    opt1, opt2, opt_empty = st.columns([1, 1, 4])
    with opt1:
        depth = st.selectbox(
            "Depth",
            options=["quick", "normal"],
            index=0 if st.session_state.depth == "quick" else 1,
            key="compare_depth_select"
        )
        st.session_state.depth = depth
    with st.form(key="compare_form"):
        col1, col2, col3 = st.columns([5, 5, 1])
        with col1:
            topic_a = st.text_input(
                label="topic_a",
                placeholder="e.g. React",
                label_visibility="collapsed"
            )
        with col2:
            topic_b = st.text_input(
                label="topic_b",
                placeholder="e.g. Vue",
                label_visibility="collapsed"
            )
        with col3:
            submitted = st.form_submit_button("vs →", use_container_width=True)
    if submitted and topic_a.strip() and topic_b.strip():
        run_comparison_and_store(
            topic_a.strip(),
            topic_b.strip(),
            st.session_state.depth
        )
    elif submitted:
        st.warning("Enter both topics to compare.")

# display report
if st.session_state.current_report:
    result = st.session_state.current_report

    st.divider()
    st.markdown(result["report"])

    # export section
    st.markdown(
        '<div class="export-label">Export</div>',
        unsafe_allow_html=True
    )
    exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 4])
    with exp_col1:
        pdf_bytes = generate_pdf(result["topic"], result["report"])
        st.download_button(
            label="↓ PDF",
            data=pdf_bytes,
            file_name=f"distill-{result['topic'][:30].replace(' ', '-').lower()}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with exp_col2:
        word_bytes = generate_word(result["topic"], result["report"])
        st.download_button(
            label="↓ Word",
            data=word_bytes,
            file_name=f"distill-{result['topic'][:30].replace(' ', '-').lower()}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )

    # follow-up questions
    if result.get("followups"):
        st.divider()
        st.markdown(
            '<div class="followup-label">Explore further</div>',
            unsafe_allow_html=True
        )
        for q in result["followups"]:
            st.markdown(f"**{q}**")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Quick answer", key=f"quick_{q}"):
                    st.session_state[f"quick_answer_{q}"] = "loading"
            with col2:
                if st.button("Full report →", key=f"full_{q}"):
                    st.session_state.prefill_topic = q
                    st.session_state.research_mode = "topic"
                    st.session_state.current_report = None
                    st.rerun()

            if st.session_state.get(f"quick_answer_{q}") == "loading":
                from agents.quick_agent import run_quick_agent
                with st.spinner("Thinking..."):
                    answer = run_quick_agent(q, result["report"])
                st.session_state[f"quick_answer_{q}"] = answer

            if (
                st.session_state.get(f"quick_answer_{q}") and
                st.session_state[f"quick_answer_{q}"] != "loading"
            ):
                st.markdown(
                    f'<div class="quick-answer">{st.session_state[f"quick_answer_{q}"]}</div>',
                    unsafe_allow_html=True
                )

    st.markdown(
        f'<div class="report-footer">Distill · {result["sources_found"]} sources · powered by Groq + Tavily</div>',
        unsafe_allow_html=True
    )
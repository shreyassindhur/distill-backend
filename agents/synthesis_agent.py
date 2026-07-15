import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODELS = [
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

REASONING_INSTRUCTION = """
CRITICAL RULES FOR QUALITY:

1. ANALYZE, DON'T SUMMARIZE
   Bad: "According to Source A, AI agents are becoming more capable."
   Good: "Capability claims from lab benchmarks consistently outpace real-world
          performance by 20-40% — suggesting the metric being measured doesn't
          reflect actual deployment conditions."

2. EVERY CLAIM NEEDS A QUALITY SIGNAL
   Label each significant claim:
   [Established] — supported by multiple independent sources
   [Emerging] — supported by recent but limited evidence
   [Contested] — actively debated with credible voices on both sides
   [Speculative] — extrapolated, not yet evidenced

3. NUMBERS NEED CONTEXT
   Bad: "The market is worth USD 4.2 billion."
   Good: "The market is worth USD 4.2 billion (Gartner, 2024) — though this figure
          includes adjacent categories, making direct year-on-year comparison unreliable."

4. SHOW THE TENSION IN THE EVIDENCE
   If sources disagree, explain WHY — not just "experts disagree" but the
   underlying reason: different data, methodology, definitions, or incentives.

5. SAY WHAT MOST COVERAGE MISSES
   Every topic has a dominant narrative and an underreported angle. Find it.

6. BE SPECIFIC ABOUT UNCERTAINTY
   If something is unknown, say so explicitly.

7. NEVER USE FILLER PHRASES
   Ban: "It is worth noting", "Importantly", "In conclusion", "To summarize"

8. CITE INLINE ALWAYS
   Every factual claim must have a markdown link: [Source Name](url)
   Never paste raw URLs. Never make up citations.

9. Write USD not $. Avoid special characters that break markdown.
"""

TOPIC_PROMPT = """You are a senior research analyst. Write a flowing, structured research
report that reads like a well-organized essay — not a checklist or bullet-point dump.
The report must tell a coherent story from beginning to end.

""" + REASONING_INSTRUCTION + """

Use EXACTLY this structure. Tag EVERY significant claim with an evidence tag:

# [Topic Title — clear and specific]

**Overview:** 2-3 sentences. What is this topic about and why does it matter now?
State the most important takeaway first. Include 1 evidence tag here.

## 1. Background and Context

Start from the beginning. What is the origin or foundation of this topic?
What key concepts does the reader need to understand first?
What historical developments led to the current situation?
Write 3-4 paragraphs. Every paragraph must have at least 1 evidence-tagged claim.

## 2. Current Landscape

What is the present state of this topic? Cover:
- What is known with confidence [Established] (cite sources)
- What is actively developing or changing [Emerging]
- Key players, technologies, policies, or trends
- Recent developments with dates where possible
Write 4-5 paragraphs. Every paragraph must have at least 1 evidence-tagged claim.
Use ALL FOUR tags across this section: [Established], [Emerging], [Contested], [Speculative].

## 3. Key Developments and Trends

What are the most significant recent developments?
What direction is this heading? Cover:
- Major breakthroughs or shifts
- Emerging patterns and where they point
- Differing viewpoints and why they exist [Contested]
Write 3-4 paragraphs. Every paragraph must have at least 1 evidence-tagged claim.

## 4. Challenges and Considerations

What problems, limitations, or debates exist?
What should the reader be cautious about?
Write 2-3 paragraphs. Tag each debate with [Contested].

## 5. Future Outlook

Where is this headed? What comes next?
What are the implications — for whom, on what timeline?
Write 2-3 paragraphs. Tag forward-looking claims with [Emerging] or [Speculative].

## Sources
- [Publication](url) — one sentence on what this source contributed

## Contested Claims
If sources disagree substantially:
- **Claim:** [assertion]
  **Evidence for:** [with link]
  **Evidence against:** [with link]
  **Why they diverge:** [actual reason]

If broadly consistent: "Sources are broadly consistent on this topic."

## Follow-up Questions
- Question one?
- Question two?
- Question three?

CRITICAL RULES:
- ALWAYS tag every significant claim with [Established], [Emerging], [Contested], or [Speculative]
- The tags are what make this report valuable — do not skip them
- Place tags inline right after the claim: "Remote work boosts productivity by 13% [Established] — [Source](url)"
- Every section must flow naturally into the next
- Write USD not $. Avoid special characters that break markdown.
- Every factual claim needs a markdown link: [Source](url)
- Never use filler phrases like "It is worth noting", "In conclusion", "To summarize"
"""

URL_PROMPT = """You are a senior research analyst specializing in source criticism.
The user shared an article — analyze it in a flowing, readable format.

""" + REASONING_INSTRUCTION + """

Use EXACTLY this structure:

# Article Analysis: [Title]

**Overview:** One sentence — what is this article's core argument and is it supported?

## 1. What This Article Says
2-3 sentences summarizing the main argument and evidence presented.

## 2. Broader Context
3-4 paragraphs. What does the reader need to know to understand this topic fully?
Background, key concepts, and where this article fits in the larger conversation.

## 3. Assessment of Key Claims
For each major claim, assess in flowing prose:
- What the article says
- Whether it holds up [Confirmed / Overstated / Missing context]
- What other sources say [Source](url)
Cover 3-4 claims in narrative form, not as a list.

## 4. What This Article Gets Right
1-2 things the article handles well.

## 5. What This Article Misses
2-3 specific gaps or blind spots with citations.

## 6. Verdict
One paragraph — is the core argument sound, and what should the reader take away?

## Sources
- [Source](url) — contribution and limitations

## Contested Claims
- **Claim:** [assertion]
  **Article says:** [position]
  **Other sources say:** [counter with link]
  **Why they diverge:** [reason]

## Follow-up Questions
- Question one?
- Question two?
- Question three?
"""

PDF_PROMPT = """You are a senior research analyst. Analyze the uploaded document in a
flowing, readable format — situate it, assess it, and connect it to current knowledge.

""" + REASONING_INSTRUCTION + """

Use EXACTLY this structure:

# Document Analysis: [Title]

**Overview:** One sentence — what does this document argue and how does it hold up today?

## 1. What This Document Argues
2-3 sentences on the core thesis. When was it written and how does timing affect relevance?

## 2. Context and Background
What does the reader need to know to understand this document?
What was happening when it was written? What has changed since?

## 3. Assessment of Key Claims
For each major claim:
- State it precisely
- Assess: [Confirmed / Outdated / Contested / Unsupported]
- Compare with current evidence [Source](url)
Cover 4-5 claims in flowing prose.

## 4. What Has Changed
What has happened since this document was created that it couldn't account for?

## 5. Gaps and Blind Spots
2-3 things the document doesn't address or gets wrong.

## 6. How to Use This Document
What is it still useful for? Where should the reader seek more current sources?

## Sources
- [Source](url) — contribution and limitations

## Contested Claims
- **Claim:** [assertion]
  **Document says:** [position]
  **Current evidence:** [counter with link]
  **Why they diverge:** [reason]

## Follow-up Questions
- Question one?
- Question two?
- Question three?
"""

COMPARISON_PROMPT = """You are a senior research analyst producing a comparative analysis.
Reach a defensible conclusion based on evidence — not neutral description.

""" + REASONING_INSTRUCTION + """

# [Topic A] vs [Topic B]: [Sharp framing]

One sentence verdict: how do these compare and for whom?

## What's Actually Being Compared
2-3 sentences. Identify any category errors.

## The Evidence on [Topic A]
3-4 sentences with sources. Label claims [Established]/[Contested]/[Emerging].

## The Evidence on [Topic B]
Same structure.

## Head to Head
| Dimension | [Topic A] | [Topic B] |
|-----------|-----------|-----------|
| [Dimension 1] | [evidence-backed] | [evidence-backed] |
| [Dimension 2] | ... | ... |
| [Dimension 3] | ... | ... |
| [Dimension 4] | ... | ... |
| [Dimension 5] | ... | ... |

## Where the Comparison Gets Complicated
1-2 paragraphs. What does "better" actually depend on?

## When to Choose [Topic A]
3 specific scenarios with reasoning.

## When to Choose [Topic B]
Same structure.

## Verdict
One paragraph. Which is better, under what conditions, for whom.

## Sources
- [Source](url) — [type] — contribution and limitations

## Contested Claims
- **Claim:** [disputed comparison]
  **Evidence for A:** [with link]
  **Evidence for B:** [with link]
  **Why they diverge:** [reason]

## Follow-up Questions
- Question one?
- Question two?
- Question three?
"""

WRITE_PAPER_PROMPT = """You are an expert academic researcher writing a full IEEE-format 
research paper. The user provides a topic and web sources. Produce a complete, 
publishable-quality paper in IEEE conference paper structure.

CRITICAL RULES:
1. Write a REAL paper — original analysis and argument, not a summary of sources
2. Every factual claim MUST have an inline citation [Author/Source](url)
3. Use precise, formal academic language throughout
4. The paper must have a clear, arguable thesis — not just "this paper reviews X"
5. Acknowledge limitations and counterarguments explicitly
6. Write USD not $. No special characters.
7. Citations: [Author et al., Year](url) or [Publication Name, Year](url)
8. Main sections use ## (H2), subsections use ### (H3)
9. Every section must be substantive — minimum 2 paragraphs each

Output in EXACTLY this structure:

# [Paper Title — specific and descriptive, not generic]

**Abstract**
150-200 words. Must contain: (1) motivation/problem, (2) objective,
(3) methodology, (4) key findings, (5) conclusion/contribution.
Written in past tense. No citations in abstract.

**Keywords:** keyword1, keyword2, keyword3, keyword4, keyword5

---

## I. Introduction

Open with the broader problem context — why does this topic matter now?
State the gap in knowledge or practice this paper addresses.
Clearly state the thesis: "This paper argues that..." 
Outline the paper structure.
State scope and limitations.
[3 paragraphs, cited throughout]

## II. Related Work

Synthesize existing literature — do NOT list papers one by one.
Group thematically: what has been studied, methods used, what found.
Identify where literature agrees, diverges, and what gap remains.
Position this paper: "Unlike X, this paper..."
[3 paragraphs, heavily cited]

## III. Methodology

Describe the research approach: systematic literature synthesis,
source selection criteria, analytical framework.
Be specific about what sources were included and why.
Acknowledge limitations of the methodology.
[2 paragraphs]

## IV. Results and Analysis

### IV-A. [First Key Dimension — name it specifically]
Present evidence, analyze it, draw conclusions.
What does this evidence mean? Why does it matter?
Address counterarguments.
[2-3 paragraphs, cited]

### IV-B. [Second Key Dimension]
Same structure.
[2-3 paragraphs, cited]

### IV-C. [Third Key Dimension]
Same structure. Address the strongest counterargument to your thesis here.
[2-3 paragraphs, cited]

## V. Discussion

Synthesize findings — what do they mean together?
How do they answer the research question from the introduction?
Theoretical implications.
Practical implications — for whom, on what timeline.
Limitations that future research should address.
[3 paragraphs]

## VI. Conclusion

Restate the research question and how this paper answered it.
Key contributions — what does this paper add?
2-3 specific directions for future research.
Closing sentence on significance.
[1-2 paragraphs]

## References

List ALL sources cited. Format each as:
[N] Author/Publication (Year). Title. Source. [URL](url)

Number sequentially. Include every source cited in the paper body.
"""

IMPROVE_PAPER_PROMPT = """You are an expert academic editor and research methodologist.
The user uploaded a research paper or draft. Provide specific, actionable improvements —
not generic advice but precise edits with justification.

RULES:
- Be specific. Quote the paper's actual text when critiquing.
- Suggest exact replacement language where possible.
- Every suggested improvement must be justified.
- Use current web sources to strengthen weak claims [Source](url).
- Write USD not $. No special characters.

Output in this exact format:

# Review: [Paper Title or Topic]

**Overall Assessment**
2-3 sentences: what is this paper doing well and what is its most critical weakness?
Be direct — honest feedback, not encouragement.

---

## Thesis and Argument
**Current thesis:** [quote or paraphrase the thesis]
**Assessment:** [Strong / Needs sharpening / Unclear]
**Suggested revision:** [if needed, write a stronger version]
**Why:** [specific reason]

## Structure and Flow
Assess the overall structure. Does each section do what it should?
For each section with a problem:
- **[Section name]:** [specific issue] → [specific fix]

## Evidence and Citations
List the 3-4 most significant evidential weaknesses:
- **Weak claim:** "[quote from paper]"
  **Problem:** [why unsupported or overstated]
  **Stronger version:** [rewritten with better evidence]
  **Supporting source:** [Source](url)

## What's Missing
2-3 specific gaps:
- **Missing:** [what's absent]
  **Why it matters:** [how it strengthens the paper]
  **Suggested source:** [Source](url)

## Language and Clarity
2-3 passages that need rewriting:
- **Original:** "[quote]"
  **Problem:** [vague / jargon / passive / unclear]
  **Revised:** "[rewritten version]"

## Counterarguments Not Addressed
Strongest objections to this paper's thesis it doesn't engage with:
- **Objection:** [the counterargument]
  **How to address it:** [specific suggestion]

## Strengthened Abstract
Rewrite the abstract (or write one if missing) incorporating improvements.
150-200 words in proper IEEE academic format.

## Priority Action List
5 most important changes, ranked:
1. [Most critical — do this first]
2.
3.
4.
5.
"""

def synthesize_thread_topics(thread: list) -> str:
    topics_text = "\n\n".join([
        f"Topic: {t['topic']}\nSummary: {t['report']}"
        for t in thread
    ])
    messages = [
        {
            "role": "system",
            "content": """You are a research synthesis agent. Find non-obvious connections
between the user's research topics — specifically how they connect and what that suggests.

Format exactly:

## Your Research Thread
One sentence on the common theme — specific, not generic.

## How These Connect
2-3 sentences on the actual intellectual connection.

## What This Suggests
One specific next step or insight they probably haven't thought of.

Maximum 150 words. Be specific. No filler."""
        },
        {
            "role": "user",
            "content": f"Topics researched:\n\n{topics_text}\n\nFind the non-obvious connections."
        }
    ]
    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=300,
                temperature=0.4,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Distill] Thread {model} failed: {e}")
            continue
    return ""

def run_synthesis_agent(
    topic: str,
    search_results: list,
    mode: str = "topic",
    tone: str = "default"
) -> str:

    # Separate academic (Semantic Scholar) from web sources
    academic = [r for r in search_results if r.get("source") == "semantic_scholar"]
    web = [r for r in search_results if r.get("source") != "semantic_scholar"]

    formatted_results = ""

    # Academic sources first — clearly labeled
    if academic:
        formatted_results += "\n=== PEER-REVIEWED ACADEMIC SOURCES ===\n"
        formatted_results += "These are real papers from Semantic Scholar with verified metadata.\n"
        formatted_results += "Cite these with author names and years: [Author et al., Year](url)\n\n"
        for i, r in enumerate(academic, 1):
            formatted_results += f"[Academic Source {i}]\n"
            formatted_results += f"Title: {r.get('title', 'Unknown')}\n"
            formatted_results += f"Authors: {r.get('authors', 'Unknown')}\n"
            formatted_results += f"Year: {r.get('year', 'n.d.')}\n"
            formatted_results += f"Venue: {r.get('venue', 'Unknown')}\n"
            formatted_results += f"Citations: {r.get('citation_count', 0)}\n"
            formatted_results += f"URL: {r.get('url', '')}\n"
            formatted_results += f"Content: {r.get('content', '')}\n"
            formatted_results += "---\n"

    # Web sources second
    if web:
        formatted_results += "\n=== WEB SOURCES ===\n"
        formatted_results += "Current web sources for recent context and industry data.\n\n"
        for i, r in enumerate(web, 1):
            if "error" not in r:
                formatted_results += f"[Web Source {i}]\n"
                formatted_results += f"Title: {r.get('title', 'Unknown')}\n"
                formatted_results += f"URL: {r.get('url', '')}\n"
                formatted_results += f"Content: {r.get('content', '')}\n"
                formatted_results += "---\n"

    tone_instructions = {
        "default":    "Clear, direct, analytical. Explain technical terms simply. No jargon. No filler.",
        "academic":   "Precise academic language. Acknowledge methodological limitations. Cite rigorously.",
        "executive":  "Lead with conclusion. Short paragraphs. Decision-focused.",
        "journalist": "Lead with most newsworthy angle. Active voice. Name names.",
    }
    tone_text = tone_instructions.get(tone, tone_instructions["default"])

    prompt_map = {
        "url":           URL_PROMPT,
        "pdf":           PDF_PROMPT,
        "comparison":    COMPARISON_PROMPT,
        "write_paper":   WRITE_PAPER_PROMPT,
        "improve_paper": IMPROVE_PAPER_PROMPT,
    }
    system_prompt = prompt_map.get(mode, TOPIC_PROMPT)
    is_paper = mode in ("write_paper", "improve_paper")

    # Extra instruction for paper mode with academic sources
    academic_instruction = ""
    if academic and is_paper:
        academic_instruction = f"""
IMPORTANT: You have {len(academic)} real peer-reviewed papers from Semantic Scholar.
- Use these as your PRIMARY citations — they have real authors, years, and DOIs
- Cite them as: [First Author et al., Year](url) or [Author, Year](url) for single author
- These are VERIFIED — do not change author names or years
- Prefer high citation-count papers for established claims
- Use web sources only for recent context not covered by academic papers
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""Topic: {topic}

Tone: {tone_text}
{academic_instruction}
Sources ({len(search_results)} total — {len(academic)} academic, {len(web)} web):
{formatted_results}

Write the full {"paper" if is_paper else "report"} now.
Be analytical, not descriptive.
Every factual claim needs a citation.
Prefer academic sources for established claims.
Every section needs substance — no padding."""
        }
    ]

    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=4000 if is_paper else 3000,
                temperature=0.2 if is_paper else 0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Distill] Model {model} failed: {e}")
            continue

    return "Unable to generate at this time. Please try again."
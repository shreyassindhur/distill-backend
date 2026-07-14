from datetime import datetime
import re


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def clean_markdown(text: str) -> str:
    """Remove markdown syntax for LaTeX conversion."""
    # Remove markdown links but keep text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Bold
    text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", text)
    # Italic
    text = re.sub(r"\*([^*]+)\*", r"\\textit{\1}", text)
    # Remove citation labels like [Established] [Contested]
    text = re.sub(
        r"\[(Established|Emerging|Contested|Speculative|Confirmed|Overstated|Missing context|Outdated|Unsupported)\]",
        "",
        text,
    )
    return text.strip()


def markdown_to_latex_body(markdown: str) -> tuple[str, list[str]]:
    """Convert markdown report to LaTeX body text.

    Returns (latex_body, references_list).
    """
    lines = markdown.split("\n")
    latex_lines = []
    references = []
    ref_map = {}  # url -> citation number
    in_references = False
    in_itemize = False
    in_enumerate = False
    skip_next = False

    def get_ref_num(url: str, label: str) -> str:
        """Get or create citation number for a URL."""
        if url not in ref_map:
            ref_map[url] = len(ref_map) + 1
            # Store reference
            references.append((ref_map[url], label, url))
        return str(ref_map[url])

    def process_inline(text: str) -> str:
        """Process inline markdown to LaTeX."""

        # Convert [Label](url) to \cite{refN}
        def replace_link(m):
            label = m.group(1)
            url = m.group(2)
            num = get_ref_num(url, label)
            return f"\\cite{{ref{num}}}"

        text = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", replace_link, text)
        # Bold
        text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", text)
        # Italic
        text = re.sub(r"\*([^*]+)\*", r"\\textit{\1}", text)
        # Remove quality labels
        text = re.sub(
            r"\[(Established|Emerging|Contested|Speculative|Confirmed|Overstated|Missing context|Outdated|Unsupported)\]",
            "",
            text,
        )
        # Escape remaining special chars (but not already converted LaTeX)
        text = (
            text.replace("&", "\\&")
            .replace("%", "\\%")
            .replace("$", "\\$")
            .replace("#", "\\#")
            .replace("~", "\\~{}")
        )
        return text.strip()

    def close_list():
        nonlocal in_itemize, in_enumerate
        if in_itemize:
            latex_lines.append("\\end{itemize}")
            in_itemize = False
        if in_enumerate:
            latex_lines.append("\\end{enumerate}")
            in_enumerate = False

    section_count = 0

    for i, raw in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        line = raw.strip()

        # Detect references section
        if re.match(
            r"^#+\s*(References|Bibliography|Works Cited)",
            line,
            re.IGNORECASE,
        ):
            close_list()
            in_references = True
            continue

        if in_references:
            # Parse reference lines like [1] Author (Year). Title. Source. [URL](url)
            ref_match = re.match(r"^\[(\d+)\]\s*(.+)", line)
            if ref_match:
                ref_num = ref_match.group(1)
                ref_text = ref_match.group(2)
                # Extract URL if present
                url_match = re.search(r"\[([^\]]*)\]\(([^\)]+)\)", ref_text)
                url = url_match.group(2) if url_match else ""
                clean_text = re.sub(r"\[([^\]]*)\]\(([^\)]+)\)", r"\1", ref_text)
                clean_text = escape_latex(clean_text)
                if url not in ref_map:
                    ref_map[url] = int(ref_num)
                    references.append((int(ref_num), clean_text, url))
            continue

        # Skip H1 title (goes in \title{})
        if line.startswith("# ") and section_count == 0 and i < 5:
            continue

        # H2 — IEEE section
        if line.startswith("## "):
            close_list()
            section_count += 1
            title = line[3:].strip()
            # Remove roman numerals if already present
            title = re.sub(r"^[IVX]+\.\s*", '', title)
            clean_title = escape_latex(title)
            latex_lines.append(f"\n\\section{{{clean_title}}}")
            continue

        # H3 — subsection
        if line.startswith("### "):
            close_list()
            title = line[4:].strip()
            title = re.sub(r"^\d+\.\d+\s*", '', title)
            clean_title = escape_latex(title)
            latex_lines.append(f"\n\\subsection{{{clean_title}}}")
            continue

        # H4 — subsubsection
        if line.startswith("#### "):
            close_list()
            title = escape_latex(line[5:].strip())
            latex_lines.append(f"\n\\subsubsection{{{title}}}")
            continue

        # Bullet list
        if line.startswith("- ") or line.startswith("* "):
            if not in_itemize:
                close_list()
                latex_lines.append("\\begin{itemize}")
                in_itemize = True
            content = process_inline(line[2:].strip())
            latex_lines.append(f"  \\item {content}")
            continue

        # Numbered list
        if re.match(r"^\d+\.\s", line):
            if not in_enumerate:
                close_list()
                latex_lines.append("\\begin{enumerate}")
                in_enumerate = True
            content = process_inline(re.sub(r"^\d+\.\s", '', line))
            latex_lines.append(f"  \\item {content}")
            continue

        # Table
        if line.startswith("|"):
            close_list()
            cells = [c.strip() for c in line.strip("|").split("|")]
            # Check if separator row
            if all(re.match(r"^-+$", c) for c in cells if c):
                continue
            # Check if header (next line is separator)
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            is_header = bool(re.match(r"^\|[-\s|]+\|$", next_line))
            if is_header:
                cols = len(cells)
                col_spec = " | ".join(["l"] * cols)
                latex_lines.append(f"\\begin{{table}}[h]")
                latex_lines.append(f"\\centering")
                latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
                latex_lines.append("\\hline")
                row = " & ".join([escape_latex(process_inline(c)) for c in cells])
                # FIX: Handled string replacement prior to f-string structure
                bold_row = row.replace(" & ", "} & \\textbf{")
                latex_lines.append(f"\\textbf{{{bold_row}}} \\\\")
                latex_lines.append("\\hline")
            else:
                row = " & ".join([escape_latex(process_inline(c)) for c in cells])
                latex_lines.append(f"{row} \\\\")
                # Check if this is the last table row
                next_line2 = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if not next_line2.startswith("|"):
                    latex_lines.append("\\hline")
                    latex_lines.append("\\end{tabular}")
                    latex_lines.append("\\caption{Comparison}")
                    latex_lines.append("\\end{table}")
            continue

        # Horizontal rule
        if re.match(r"^[-_*]{3,}$", line):
            close_list()
            continue

        # Blockquote
        if line.startswith(">"):
            close_list()
            content = process_inline(line[1:].strip())
            latex_lines.append(f"\\begin{{quote}}\n{content}\n\\end{{quote}}")
            continue

        # Empty line
        if not line:
            close_list()
            latex_lines.append("")
            continue

        # Regular paragraph
        close_list()
        content = process_inline(line)
        if content:
            latex_lines.append(content)

    close_list()

    return "\n".join(latex_lines), references


def generate_latex(
    topic: str, report: str, authors: str = "Distill Research Assistant"
) -> str:
    """Generate a complete IEEE-formatted LaTeX document.

    Returns the full .tex file content as a string.
    """

    # Extract title from first H1 in report
    title = topic
    for line in report.split("\n"):
        if line.startswith("# "):
            title = line[2:].strip()
            break

    # Extract abstract if present
    abstract = ""
    abstract_match = re.search(
        r"\*\*Abstract\*\*\s*\n(.*?)(?=\n\*\*Keywords|\n##|\Z)",
        report,
        re.DOTALL | re.IGNORECASE,
    )
    if not abstract_match:
        abstract_match = re.search(
            r"## Abstract\s*\n(.*?)(?=\n##|\Z)", report, re.DOTALL | re.IGNORECASE
        )
    if abstract_match:
        abstract = abstract_match.group(1).strip()
        abstract = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", abstract)
        abstract = re.sub(r"\*\*([^*]+)\*\*", r"\1", abstract)
        abstract = (
            abstract.replace("&", "\\&")
            .replace("%", "\\%")
            .replace("$", "\\$")
            .replace("#", "\\#")
        )

    # Extract keywords
    keywords = "research, analysis, literature review"
    kw_match = re.search(
        r"\*\*Keywords[:\*]*\*?\*?\s*([^\n]+)", report, re.IGNORECASE
    )
    if kw_match:
        keywords = kw_match.group(1).strip().strip("*").strip()

    # Convert body
    body_latex, references = markdown_to_latex_body(report)

    # Build bibliography
    bib_entries = []
    for num, label, url in sorted(references, key=lambda x: x[0]):
        clean_label = escape_latex(label)
        clean_url = url.replace("_", "\\_").replace("%", "\\%")
        bib_entries.append(
            f"\\bibitem{{ref{num}}}\n"
            f"{clean_label}.\n"
            f"\\newblock Available: \\url{{{url}}}"
        )

    # If no references extracted, add placeholder
    if not bib_entries:
        bib_entries = [
            "\\bibitem{ref1}\nSources retrieved via Distill research pipeline.\n\\newblock Powered by Tavily Search API."
        ]

    bib_block = "\n\n".join(bib_entries)

    # Clean title for LaTeX
    clean_title = escape_latex(title)
    year = datetime.now().year

    latex = f"""\\documentclass[conference]{{IEEEtran}}
\\IEEEoverridecommandlockouts

\\usepackage{{cite}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage{{algorithmic}}
\\usepackage{{graphicx}}
\\usepackage{{textcomp}}
\\usepackage{{xcolor}}
\\usepackage{{hyperref}}
\\usepackage{{url}}

\\hypersetup{{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,
    urlcolor=blue,
    citecolor=blue
}}

\\def\\BibTeX{{{{\\rm B\\kern-.05em{{\\sc i\\kern-.025em b}}\\kern-.08em
    T\\kern-.1667em\\lower.7ex\\hbox{{E}}\\kern-.125emX}}}}

\\begin{{document}}

\\title{{{clean_title}}}

\\author{{\\IEEEauthorblockN{{{escape_latex(authors)}}}
\\IEEEauthorblockA{{\\textit{{Generated by Distill}} \\\\
distill-ui.vercel.app \\\\
{year}}}
}}

\\maketitle

\\begin{{abstract}}
{abstract if abstract else "This paper presents a comprehensive analysis of " + escape_latex(topic) + ". The research synthesizes current literature and empirical evidence to provide insights into the key dimensions of this topic, including methodology, findings, and implications for future research."}
\\end{{abstract}}

\\begin{{IEEEkeywords}}
{escape_latex(keywords)}
\\end{{IEEEkeywords}}

{body_latex}

\\begin{{thebibliography}}{{99}}
{bib_block}
\\end{{thebibliography}}

\\end{{document}}
"""

    return latex
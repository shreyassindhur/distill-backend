from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
from pydantic import BaseModel
from collections import defaultdict
import uvicorn, io

import database as db
db.init_db()

from orchestrator import (
    run_research,
    run_url_research,
    run_pdf_research,
    run_comparison_research,
    run_write_paper,
    run_improve_paper,
    run_analyze,
)
from tools.pdf_export import generate_pdf
from tools.word_export import generate_word
from tools.latex_export import generate_latex

app = FastAPI(title="Distill API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

research_threads: dict = defaultdict(list)


# ── request models ────────────────────────────────────────────────────────────

class TopicRequest(BaseModel):
    topic: str
    depth: str = "normal"
    tone: str = "default"

class URLRequest(BaseModel):
    url: str
    tone: str = "default"

class CompareRequest(BaseModel):
    topic_a: str
    topic_b: str
    depth: str = "normal"

class WritePaperRequest(BaseModel):
    topic: str
    depth: str = "normal"

class ExportRequest(BaseModel):
    topic: str
    report: str
    format: str = "pdf"

class LaTeXExportRequest(BaseModel):
    topic: str
    report: str
    authors: str = "Distill Research Assistant"

class FeedbackRequest(BaseModel):
    session_id: str
    love: str = ""
    improve: str = ""
    rating: int = 0
    name: str = ""
    email: str = ""

class DeductRequest(BaseModel):
    session_id: str
    cost: int = 1

class AwardRequest(BaseModel):
    session_id: str
    amount: int = 1

class SessionRequest(BaseModel):
    session_id: str

class QuickAnswerRequest(BaseModel):
    question: str
    context: str

class ThreadAddRequest(BaseModel):
    session_id: str
    topic: str
    report: str

class ThreadSynthesizeRequest(BaseModel):
    session_id: str


# ── health ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Distill API running"}


# ── research endpoints ────────────────────────────────────────────────────────

@app.post("/research/topic")
def research_topic(req: TopicRequest):
    try:
        return JSONResponse(content=run_research(req.topic, depth=req.depth, tone=req.tone))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/research/url")
def research_url(req: URLRequest):
    try:
        return JSONResponse(content=run_url_research(req.url, tone=req.tone))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/research/compare")
def research_compare(req: CompareRequest):
    try:
        return JSONResponse(content=run_comparison_research(req.topic_a, req.topic_b, depth=req.depth))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/research/pdf")
async def research_pdf(
    file: UploadFile = File(...),
    tone: str = Form("default")
):
    try:
        await file.seek(0)
        return JSONResponse(content=run_pdf_research(file, tone=tone))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/research/analyze")
async def research_analyze(
    url: str = Form(""),
    file: UploadFile = File(None),
    tone: str = Form("default")
):
    try:
        if file:
            await file.seek(0)
        return JSONResponse(content=run_analyze(tone=tone, url=url, uploaded_file=file))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── paper endpoints ───────────────────────────────────────────────────────────

@app.post("/paper/write")
def paper_write(req: WritePaperRequest):
    try:
        return JSONResponse(content=run_write_paper(req.topic, depth=req.depth))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/paper/improve")
async def paper_improve(file: UploadFile = File(...)):
    try:
        await file.seek(0)
        return JSONResponse(content=run_improve_paper(file))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── export endpoints ──────────────────────────────────────────────────────────

@app.post("/export/pdf")
def export_pdf(req: ExportRequest):
    try:
        pdf_bytes = generate_pdf(req.topic, req.report)
        slug = req.topic[:30].replace(" ", "-").lower()
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=distill-{slug}.pdf"}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/export/word")
def export_word(req: ExportRequest):
    try:
        word_bytes = generate_word(req.topic, req.report)
        slug = req.topic[:30].replace(" ", "-").lower()
        return StreamingResponse(
            io.BytesIO(word_bytes),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename=distill-{slug}.docx"}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/export/latex")
def export_latex(req: LaTeXExportRequest):
    try:
        tex_content = generate_latex(req.topic, req.report, req.authors)
        slug = req.topic[:30].replace(" ", "-").lower()
        tex_bytes = tex_content.encode("utf-8")
        return Response(
            content=tex_bytes,
            media_type="application/x-tex",
            headers={"Content-Disposition": f"attachment; filename=distill-{slug}.tex"}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── feedback & credits ────────────────────────────────────────────────────

@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    try:
        def _gib(t: str) -> bool:
            c = [x for x in t.lower() if x.isalpha()]
            if len(c) < 4: return len(t.strip()) > 0
            v = sum(1 for x in c if x in "aeiou")
            if v == 0: return True
            if v / len(c) < 0.15: return True
            if len(set(c)) / len(c) < 0.25: return True
            run = mx = 0
            for x in c:
                if x in "aeiou": run = 0
                else: run += 1; mx = max(mx, run)
            if mx > 5: return True
            return False
        if not req.name.strip():
            return JSONResponse(status_code=400, content={"error": "Name is required"})
        if not req.email.strip() or "@" not in req.email or "." not in req.email.split("@")[-1]:
            return JSONResponse(status_code=400, content={"error": "Valid email is required"})
        texts = [t for t in [req.love, req.improve] if t.strip()]
        if texts and all(_gib(t) for t in texts) and not req.rating:
            return JSONResponse(status_code=400, content={"error": "feedback appears to be gibberish"})
        combined = f"♥ {req.love}\n▲ {req.improve}\n★ {req.rating}/5"
        db.record_feedback(req.session_id, req.love, req.improve, req.rating, combined, req.name, req.email)
        s = db.add_credits(req.session_id, 10)
        if not s: return JSONResponse(status_code=404, content={"error": "session not found"})
        return {"ok": True, "credits": s}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/credits/init")
def credits_init():
    try:
        s = db.create_session()
        return s
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/credits/{session_id}")
def credits_get(session_id: str):
    try:
        s = db.get_session(session_id)
        if not s: return JSONResponse(status_code=404, content={"error": "session not found"})
        return s
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/credits/deduct")
def credits_deduct(req: DeductRequest):
    try:
        s = db.deduct_credits(req.session_id, req.cost)
        if not s: return JSONResponse(status_code=400, content={"error": "insufficient credits or invalid session"})
        return s
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/credits/award")
def credits_award(req: AwardRequest):
    try:
        s = db.add_credits(req.session_id, req.amount)
        if not s: return JSONResponse(status_code=404, content={"error": "session not found"})
        return s
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/quick-answer")
def quick_answer(req: QuickAnswerRequest):
    try:
        from agents.quick_agent import run_quick_agent
        return {"answer": run_quick_agent(req.question, req.context)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── research thread ───────────────────────────────────────────────────────────

@app.post("/thread/add")
def add_to_thread(req: ThreadAddRequest):
    thread = research_threads[req.session_id]
    if not any(t["topic"] == req.topic for t in thread):
        thread.append({"topic": req.topic, "report": req.report[:500]})
    return {"count": len(thread), "topics": [t["topic"] for t in thread]}


@app.post("/thread/synthesize")
def synthesize_thread(req: ThreadSynthesizeRequest):
    thread = research_threads.get(req.session_id, [])
    if len(thread) < 2:
        return {"synthesis": None}
    try:
        from agents.synthesis_agent import synthesize_thread_topics
        return {"synthesis": synthesize_thread_topics(thread)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/thread/{session_id}")
def get_thread(session_id: str):
    thread = research_threads.get(session_id, [])
    return {"count": len(thread), "topics": [t["topic"] for t in thread]}


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
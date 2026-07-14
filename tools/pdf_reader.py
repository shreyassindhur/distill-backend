import io


def read_pdf(uploaded_file) -> dict:
    try:
        import PyPDF2

        # FastAPI UploadFile: read from .file (SpooledTemporaryFile)
        # Streamlit UploadedFile: read from .getvalue()
        raw = None

        if hasattr(uploaded_file, "file"):
            # FastAPI UploadFile — .file is a SpooledTemporaryFile
            uploaded_file.file.seek(0)
            raw = uploaded_file.file.read()
        elif hasattr(uploaded_file, "getvalue"):
            # Streamlit UploadedFile
            raw = uploaded_file.getvalue()
        elif hasattr(uploaded_file, "read"):
            raw = uploaded_file.read()

        if not raw:
            return {"error": "Could not read file — empty or unreadable"}

        reader = PyPDF2.PdfReader(io.BytesIO(raw))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            return {"error": "PDF appears to be scanned or image-only — no extractable text found"}

        return {
            "content": text[:4000],
            "pages": len(reader.pages)
        }

    except Exception as e:
        return {"error": str(e)}
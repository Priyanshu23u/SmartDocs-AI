from typing import List, Optional
import os
from dataclasses import dataclass
from langchain.docstore.document import Document
import fitz  # PyMuPDF

try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

from .config import settings
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedDoc:
    text: str
    pages: List[str]
    source_name: str


def parse_pdf_pymupdf(pdf_path: str) -> ParsedDoc:
    pages: List[str] = []
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            txt = page.get_text() or ""
            pages.append(txt)
    full_text = "\n".join(pages)
    return ParsedDoc(text=full_text, pages=pages, source_name=os.path.basename(pdf_path))


def parse_pdf_ocr(pdf_path: str) -> Optional[ParsedDoc]:
    if not settings.ENABLE_OCR or not OCR_AVAILABLE:
        return None
    try:
        images = convert_from_path(pdf_path)
        pages = [pytesseract.image_to_string(img) for img in images]
        full_text = "\n".join(pages)
        return ParsedDoc(text=full_text, pages=pages, source_name=os.path.basename(pdf_path))
    except Exception as e:
        logger.warning(f"OCR fallback failed: {e}")
        return None


def parse_pdf(
    pdf_path: str,
    use_llamaparse: bool = False,
    llamaparse_instruction: Optional[str] = None,
) -> ParsedDoc:
    if use_llamaparse and settings.LLAMA_PARSE_API_KEY:
        try:
            from llama_parse import LlamaParse

            instruction = (
                llamaparse_instruction
                or "Convert this PDF to markdown preserving tables. Be precise."
            )
            parser = LlamaParse(
                api_key=settings.LLAMA_PARSE_API_KEY,
                result_type="markdown",
                parsing_instruction=instruction,
                max_timeout=5000,
            )
            docs = parser.load_data(pdf_path)
            md_text = docs.text if docs else ""
            if md_text.strip():
                # Split pseudo-pages for metadata; still store entire text page-wise
                pages = md_text.split("\n\n# ")
                return ParsedDoc(
                    text=md_text, pages=pages, source_name=os.path.basename(pdf_path)
                )
        except Exception as e:
            logger.warning(f"LlamaParse failed, falling back to PyMuPDF: {e}")

    parsed = parse_pdf_pymupdf(pdf_path)
    if parsed.text.strip():
        return parsed

    ocr = parse_pdf_ocr(pdf_path)
    if ocr and ocr.text.strip():
        return ocr

    raise ValueError(
        "No extractable text found. File may be image-only and OCR is disabled or failed."
    )


def to_langchain_documents(parsed: ParsedDoc) -> List[Document]:
    docs: List[Document] = []
    for i, page_text in enumerate(parsed.pages, start=1):
        if page_text and page_text.strip():
            docs.append(
                Document(
                    page_content=page_text,
                    metadata={"source": parsed.source_name, "page_number": i},
                )
            )
    if not docs and parsed.text.strip():
        docs = [
            Document(
                page_content=parsed.text,
                metadata={"source": parsed.source_name, "page_number": 1},
            )
        ]
    return docs

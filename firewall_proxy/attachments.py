from __future__ import annotations

import base64
import binascii
import io
import mimetypes
import re
from dataclasses import asdict, dataclass, field
from typing import Any


DATA_URI_RE = re.compile(r"^data:(?P<mime>[-\w.+/]+)?;base64,(?P<data>.+)$", re.DOTALL)
IMAGE_MIME_PREFIX = "image/"
PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TEXT_MIME_PREFIX = "text/"


@dataclass(slots=True)
class ExtractedTextPage:
    page_number: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AttachmentImage:
    page_number: int | None
    media_type: str
    data: bytes = field(repr=False)


@dataclass(slots=True)
class TextChunk:
    attachment_index: int
    attachment_name: str
    media_type: str
    page_number: int | None
    start_offset: int
    end_offset: int
    text: str

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "attachment_index": self.attachment_index,
            "attachment_name": self.attachment_name,
            "media_type": self.media_type,
            "page_number": self.page_number,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "char_count": len(self.text),
        }


@dataclass(slots=True)
class AttachmentContent:
    index: int
    name: str
    media_type: str
    kind: str
    raw_bytes: bytes = field(default=b"", repr=False)
    text_pages: list[ExtractedTextPage] = field(default_factory=list)
    images: list[AttachmentImage] = field(default_factory=list, repr=False)
    extraction_quality: float = 0.0
    is_text_attachment: bool = False
    is_multimodal: bool = False
    notes: list[str] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "media_type": self.media_type,
            "kind": self.kind,
            "bytes": len(self.raw_bytes),
            "text_pages": len(self.text_pages),
            "image_count": len(self.images),
            "extraction_quality": self.extraction_quality,
            "is_text_attachment": self.is_text_attachment,
            "is_multimodal": self.is_multimodal,
            "notes": self.notes,
        }


@dataclass(slots=True)
class AttachmentExtractionResult:
    attachments: list[AttachmentContent]
    chunks: list[TextChunk]

    @property
    def has_attachments(self) -> bool:
        return bool(self.attachments)

    @property
    def has_text_attachments(self) -> bool:
        return any(attachment.is_text_attachment for attachment in self.attachments)

    @property
    def has_multimodal_attachments(self) -> bool:
        return any(attachment.is_multimodal for attachment in self.attachments)

    @property
    def images_for_lg4(self) -> list[AttachmentImage]:
        images: list[AttachmentImage] = []
        for attachment in self.attachments:
            images.extend(attachment.images)
        return images

    def summary_dict(self) -> dict[str, Any]:
        types: dict[str, int] = {}
        for attachment in self.attachments:
            types[attachment.kind] = types.get(attachment.kind, 0) + 1

        return {
            "present": self.has_attachments,
            "total": len(self.attachments),
            "types": types,
            "text_attachment_count": sum(1 for item in self.attachments if item.is_text_attachment),
            "multimodal_attachment_count": sum(1 for item in self.attachments if item.is_multimodal),
            "text_chunk_count": len(self.chunks),
            "attachments": [attachment.to_summary_dict() for attachment in self.attachments],
        }


def extract_attachments_from_body(
    body: Any,
    *,
    chunk_chars: int,
    chunk_overlap: int,
    max_pages: int,
    max_lg4_images: int,
) -> AttachmentExtractionResult:
    candidates = _find_attachment_candidates(body)
    attachments: list[AttachmentContent] = []

    for index, candidate in enumerate(candidates):
        attachment = _build_attachment(index, candidate)
        if attachment is None:
            continue
        _extract_attachment_content(attachment, max_pages=max_pages, max_lg4_images=max_lg4_images)
        attachments.append(attachment)

    chunks: list[TextChunk] = []
    for attachment in attachments:
        chunks.extend(
            chunk_attachment_text(
                attachment,
                chunk_chars=chunk_chars,
                chunk_overlap=chunk_overlap,
            )
        )

    return AttachmentExtractionResult(attachments=attachments, chunks=chunks)


def chunk_attachment_text(
    attachment: AttachmentContent,
    *,
    chunk_chars: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    if chunk_chars <= 0:
        return []
    overlap = max(0, min(chunk_overlap, chunk_chars - 1))
    chunks: list[TextChunk] = []

    for page in attachment.text_pages:
        text = page.text.strip()
        if not text:
            continue

        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(text_length, start + chunk_chars)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        attachment_index=attachment.index,
                        attachment_name=attachment.name,
                        media_type=attachment.media_type,
                        page_number=page.page_number,
                        start_offset=start,
                        end_offset=end,
                        text=chunk_text,
                    )
                )
            if end >= text_length:
                break
            start = max(end - overlap, start + 1)

    return chunks


def _find_attachment_candidates(body: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def walk(value: Any, path: str, parent_key: str | None = None) -> None:
        if isinstance(value, dict):
            candidate = _candidate_from_dict(value, path, parent_key)
            if candidate is not None:
                candidates.append(candidate)

            for key, child in value.items():
                walk(child, f"{path}.{key}" if path else str(key), str(key))
            return

        if isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]", parent_key)

    walk(body, "")
    return _dedupe_candidates(candidates)


def _candidate_from_dict(value: dict[str, Any], path: str, parent_key: str | None) -> dict[str, Any] | None:
    item_type = str(value.get("type") or "").lower()
    filename = value.get("filename") or value.get("file_name") or value.get("name")
    media_type = (
        value.get("media_type")
        or value.get("mime_type")
        or value.get("mimetype")
        or value.get("content_type")
    )

    if item_type in {"input_image", "image", "image_url"} or "image_url" in value:
        image_value = value.get("image_url")
        if isinstance(image_value, dict):
            image_value = image_value.get("url")
        return {
            "path": path,
            "name": filename or f"image-{len(path)}",
            "media_type": media_type,
            "data_value": image_value or value.get("data") or value.get("url"),
            "kind_hint": "image",
        }

    data_value = (
        value.get("file_data")
        or value.get("data")
        or value.get("content_base64")
        or value.get("base64")
        or value.get("bytes")
    )
    if data_value is not None and (
        item_type in {"input_file", "file", "attachment", "document"}
        or filename is not None
        or parent_key in {"attachments", "files", "documents"}
    ):
        return {
            "path": path,
            "name": filename or f"attachment-{len(path)}",
            "media_type": media_type,
            "data_value": data_value,
            "kind_hint": item_type,
        }

    if item_type in {"input_file", "file", "attachment", "document"} or (
        filename is not None and parent_key in {"attachments", "files", "documents"}
    ):
        return {
            "path": path,
            "name": filename or f"attachment-{len(path)}",
            "media_type": media_type,
            "data_value": None,
            "kind_hint": item_type,
        }

    return None


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, int]] = set()
    deduped: list[dict[str, Any]] = []
    for candidate in candidates:
        data_value = candidate.get("data_value")
        key = (
            str(candidate.get("path") or ""),
            str(candidate.get("name") or ""),
            len(data_value) if isinstance(data_value, str) else 0,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _build_attachment(index: int, candidate: dict[str, Any]) -> AttachmentContent | None:
    name = str(candidate.get("name") or f"attachment-{index}")
    raw_bytes, data_mime, note = _decode_inline_data(candidate.get("data_value"))
    media_type = str(candidate.get("media_type") or data_mime or _guess_media_type(name) or "").lower()
    kind = _classify_kind(name=name, media_type=media_type, kind_hint=str(candidate.get("kind_hint") or ""))
    notes = [note] if note else []
    if not raw_bytes and candidate.get("data_value") is not None:
        notes.append("Inline attachment data could not be decoded.")
    if not raw_bytes and candidate.get("data_value") is None:
        notes.append("Attachment metadata was present, but no inline bytes were available to inspect locally.")

    return AttachmentContent(
        index=index,
        name=name,
        media_type=media_type or "application/octet-stream",
        kind=kind,
        raw_bytes=raw_bytes,
        notes=notes,
    )


def _decode_inline_data(value: Any) -> tuple[bytes, str | None, str | None]:
    if isinstance(value, bytes):
        return value, None, None
    if not isinstance(value, str) or not value.strip():
        return b"", None, None

    stripped = value.strip()
    match = DATA_URI_RE.match(stripped)
    if match:
        try:
            return base64.b64decode(match.group("data"), validate=True), match.group("mime"), None
        except (binascii.Error, ValueError):
            return b"", match.group("mime"), "Data URI base64 payload was invalid."

    if stripped.startswith(("http://", "https://")):
        return b"", None, "Remote attachment URLs are not fetched by the firewall."

    try:
        return base64.b64decode(stripped, validate=True), None, None
    except (binascii.Error, ValueError):
        return stripped.encode("utf-8", errors="replace"), "text/plain", "Attachment content was treated as plain text."


def _guess_media_type(name: str) -> str | None:
    guessed, _ = mimetypes.guess_type(name)
    return guessed


def _classify_kind(*, name: str, media_type: str, kind_hint: str) -> str:
    lowered_name = name.lower()
    if media_type == PDF_MIME or lowered_name.endswith(".pdf"):
        return "pdf"
    if media_type == DOCX_MIME or lowered_name.endswith(".docx"):
        return "docx"
    if media_type.startswith(TEXT_MIME_PREFIX) or lowered_name.endswith((".txt", ".md", ".csv", ".json")):
        return "text"
    if media_type.startswith(IMAGE_MIME_PREFIX) or lowered_name.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff")):
        return "image"
    if "image" in kind_hint:
        return "image"
    return "unknown"


def _extract_attachment_content(
    attachment: AttachmentContent,
    *,
    max_pages: int,
    max_lg4_images: int,
) -> None:
    if not attachment.raw_bytes:
        attachment.extraction_quality = 0.0
        if attachment.kind in {"image", "pdf"}:
            attachment.is_multimodal = True
        return

    if attachment.kind == "pdf":
        _extract_pdf(attachment, max_pages=max_pages, max_lg4_images=max_lg4_images)
    elif attachment.kind == "docx":
        _extract_docx(attachment)
    elif attachment.kind == "text":
        _extract_plain_text(attachment)
    elif attachment.kind == "image":
        _extract_image(attachment)
    else:
        attachment.notes.append("Unsupported attachment type for local text extraction.")

    text_chars = sum(len(page.text.strip()) for page in attachment.text_pages)
    attachment.is_text_attachment = text_chars > 0
    if attachment.kind == "image":
        attachment.is_multimodal = True


def _extract_pdf(attachment: AttachmentContent, *, max_pages: int, max_lg4_images: int) -> None:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        attachment.notes.append(f"pypdf unavailable for PDF text extraction: {exc}")
        _render_pdf_images(attachment, max_pages=max_pages, max_images=max_lg4_images)
        return

    pages: list[ExtractedTextPage] = []
    try:
        reader = PdfReader(io.BytesIO(attachment.raw_bytes))
        total_pages = min(len(reader.pages), max_pages)
        for index in range(total_pages):
            text = reader.pages[index].extract_text() or ""
            if text.strip():
                pages.append(ExtractedTextPage(page_number=index + 1, text=text))
    except Exception as exc:
        attachment.notes.append(f"PDF text extraction failed: {exc}")

    attachment.text_pages = pages
    inspected_pages = max(1, min(max_pages, len(pages) or 1))
    text_chars = sum(len(page.text.strip()) for page in pages)
    attachment.extraction_quality = min(1.0, text_chars / float(inspected_pages * 900))

    low_quality = attachment.extraction_quality < 0.35
    if low_quality:
        attachment.notes.append("PDF text extraction appears sparse; multimodal guard is required.")
        attachment.is_multimodal = True
        _render_pdf_images(attachment, max_pages=max_pages, max_images=max_lg4_images)


def _render_pdf_images(attachment: AttachmentContent, *, max_pages: int, max_images: int) -> None:
    if max_images <= 0:
        return
    try:
        import fitz
    except Exception as exc:
        attachment.notes.append(f"PyMuPDF unavailable for PDF page rendering: {exc}")
        return

    try:
        document = fitz.open(stream=attachment.raw_bytes, filetype="pdf")
        page_count = min(len(document), max_pages, max_images)
        for index in range(page_count):
            page = document.load_page(index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            attachment.images.append(
                AttachmentImage(
                    page_number=index + 1,
                    media_type="image/png",
                    data=pixmap.tobytes("png"),
                )
            )
    except Exception as exc:
        attachment.notes.append(f"PDF page rendering failed: {exc}")


def _extract_docx(attachment: AttachmentContent) -> None:
    try:
        from docx import Document
    except Exception as exc:
        attachment.notes.append(f"python-docx unavailable for DOCX extraction: {exc}")
        return

    try:
        document = Document(io.BytesIO(attachment.raw_bytes))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
    except Exception as exc:
        attachment.notes.append(f"DOCX text extraction failed: {exc}")
        return

    if text.strip():
        attachment.text_pages = [ExtractedTextPage(page_number=1, text=text)]
        attachment.extraction_quality = min(1.0, len(text) / 2000.0)


def _extract_plain_text(attachment: AttachmentContent) -> None:
    text = attachment.raw_bytes.decode("utf-8", errors="replace")
    if text.strip():
        attachment.text_pages = [ExtractedTextPage(page_number=1, text=text)]
        attachment.extraction_quality = 1.0


def _extract_image(attachment: AttachmentContent) -> None:
    try:
        from PIL import Image
    except Exception as exc:
        attachment.notes.append(f"Pillow unavailable for image validation: {exc}")
        attachment.images.append(
            AttachmentImage(page_number=None, media_type=attachment.media_type, data=attachment.raw_bytes)
        )
        attachment.extraction_quality = 0.0
        return

    try:
        with Image.open(io.BytesIO(attachment.raw_bytes)) as image:
            output = io.BytesIO()
            image.convert("RGB").save(output, format="PNG")
            attachment.images.append(
                AttachmentImage(page_number=None, media_type="image/png", data=output.getvalue())
            )
    except Exception as exc:
        attachment.notes.append(f"Image decode failed: {exc}")
        attachment.images.append(
            AttachmentImage(page_number=None, media_type=attachment.media_type, data=attachment.raw_bytes)
        )
    attachment.extraction_quality = 0.0

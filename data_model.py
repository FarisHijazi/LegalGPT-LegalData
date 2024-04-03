"""
data model for the following JSON format:


{
    "preprocess_script_git_hash": "1234567890abcdef", // the git hash of the script that generated this file
    "schema_version": "1.0",
    "source_entity": "ministry of justice", // جهة الإصدار
    "origin_url": "http://example.com",
    "serial_number": "123456",
    "original_file_path": "path/to/file.pdf",
    "document_type": "regulation",      // نوع الوثيقة
    "circular_topic": "category",       // موضوع التعميم
    "circular_number": "123",           // رقم التعميم
    "title": "Legal Document Title",
    "issue_date": "2022-11-01",
    "effective_date": "2022-12-01",
    "expiration_date": "2023-12-31",
    "confidentiality": "public",
    "languages": ["ar", "en"],
    "contents": [
        // these are the extracted texts from the PDF
        {"text": "extracted text", "page": 1, "section": "Introduction", "text_type": "paragraph", "language": "ar"},
        {"text": "extracted text", "page": 2, "section": "Section 1", "text_type": "bullet_point", "language": "ar"}
    ],
}
"""

from pydantic import BaseModel


class OCRResponse(BaseModel):
    full_text: str

    class Page(BaseModel):
        class Block(BaseModel):
            lines: list[str]

        blocks: list[Block]

    pages: list[Page]


class LegalDocument(BaseModel):
    preprocess_script_git_hash: str
    schema_version: str
    source_entity: str
    origin_url: str
    serial_number: str
    original_file_path: str
    document_type: str
    circular_topic: str
    circular_number: str
    title: str
    issue_date: str
    effective_date: str
    expiration_date: str
    confidentiality: str
    languages: list[str]

    class Content(BaseModel):
        text: str
        page: int
        section: str
        text_type: str
        language: str

    contents: list[Content]

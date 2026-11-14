"""
PDF Text Extraction & Document Chunking Module
==============================================
Uses PyMuPDF (fitz) for fast, high-accuracy text extraction with Vietnamese Unicode support.
Splits PDF documents into semantically coherent chunks (page-by-page or section-based).
"""

import os
import re
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional


def clean_extracted_text(text: str) -> str:
    """
    Cleans extracted PDF text: normalizes whitespace, removes hyphenation at line breaks.
    """
    if not text:
        return ""
    # Remove hyphenation at line breaks: "thuật-\ntoán" -> "thuật toán"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Replace multiple newlines with double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace multiple horizontal spaces with single space
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_chunks_from_pdf(
    file_path_or_bytes,
    filename: str = "document.pdf",
    chunk_mode: str = "page",
    category: str = "Tài liệu PDF",
    min_chars_per_page: int = 40,
    max_chars_per_chunk: int = 1500
) -> List[Dict[str, Any]]:
    """
    Extracts text from PDF and produces structured document chunks.
    
    Args:
        file_path_or_bytes: Path to PDF or bytes
        filename: Name of the uploaded file
        chunk_mode: "page" (one chunk per page) or "single" (entire document)
        category: User-defined category
        min_chars_per_page: Minimum characters to consider a page valid
        max_chars_per_chunk: Maximum characters before splitting a page into sub-chunks
        
    Returns:
        List of document dictionaries ready for DocumentStore and HNSW indexing.
    """
    if isinstance(file_path_or_bytes, (str, os.PathLike)):
        doc = fitz.open(file_path_or_bytes)
    else:
        doc = fitz.open(stream=file_path_or_bytes, filetype="pdf")

    total_pages = len(doc)
    doc_title = doc.metadata.get("title") or os.path.splitext(filename)[0]
    doc_title = doc_title.replace("_", " ").replace("-", " ").strip()

    chunks: List[Dict[str, Any]] = []

    if chunk_mode == "single":
        # Combine all pages into one document
        full_text_parts = []
        for page_idx in range(total_pages):
            page_text = clean_extracted_text(doc[page_idx].get_text("text"))
            if len(page_text) >= min_chars_per_page:
                full_text_parts.append(f"--- Trang {page_idx + 1} ---\n{page_text}")
        
        full_content = "\n\n".join(full_text_parts)
        if full_content.strip():
            summary = full_content[:200].replace("\n", " ").strip() + "..."
            chunks.append({
                "title": f"{doc_title} (Toàn văn)",
                "category": category,
                "tags": ["PDF", doc_title, f"{total_pages} trang"],
                "summary": summary,
                "content": full_content,
                "page_num": 1,
                "total_pages": total_pages,
                "source_file": filename
            })

    else:
        # Page-by-page extraction (recommended for accurate search retrieval)
        for page_idx in range(total_pages):
            page = doc[page_idx]
            raw_text = page.get_text("text")
            cleaned_text = clean_extracted_text(raw_text)

            if len(cleaned_text) < min_chars_per_page:
                continue

            # If page text is very long, split into paragraphs
            if len(cleaned_text) > max_chars_per_chunk:
                paragraphs = cleaned_text.split("\n\n")
                current_chunk = []
                current_len = 0
                part_idx = 1

                for p in paragraphs:
                    p = p.strip()
                    if not p:
                        continue
                    if current_len + len(p) > max_chars_per_chunk and current_chunk:
                        chunk_text = "\n\n".join(current_chunk)
                        summary = chunk_text[:160].replace("\n", " ").strip() + "..."
                        chunks.append({
                            "title": f"{doc_title} - Trang {page_idx + 1} (Phần {part_idx})",
                            "category": category,
                            "tags": ["PDF", doc_title, f"Trang {page_idx + 1}"],
                            "summary": summary,
                            "content": chunk_text,
                            "page_num": page_idx + 1,
                            "total_pages": total_pages,
                            "source_file": filename
                        })
                        current_chunk = [p]
                        current_len = len(p)
                        part_idx += 1
                    else:
                        current_chunk.append(p)
                        current_len += len(p)

                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    summary = chunk_text[:160].replace("\n", " ").strip() + "..."
                    chunks.append({
                        "title": f"{doc_title} - Trang {page_idx + 1}" + (f" (Phần {part_idx})" if part_idx > 1 else ""),
                        "category": category,
                        "tags": ["PDF", doc_title, f"Trang {page_idx + 1}"],
                        "summary": summary,
                        "content": chunk_text,
                        "page_num": page_idx + 1,
                        "total_pages": total_pages,
                        "source_file": filename
                    })
            else:
                summary = cleaned_text[:160].replace("\n", " ").strip() + "..."
                chunks.append({
                    "title": f"{doc_title} - Trang {page_idx + 1}",
                    "category": category,
                    "tags": ["PDF", doc_title, f"Trang {page_idx + 1}"],
                    "summary": summary,
                    "content": cleaned_text,
                    "page_num": page_idx + 1,
                    "total_pages": total_pages,
                    "source_file": filename
                })

    doc.close()
    return chunks

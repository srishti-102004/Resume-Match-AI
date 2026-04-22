"""
Resume Parser — Multi-format text extraction
Supports PDF, DOCX, and plain text uploads.
"""

import io
import re


class ResumeParser:
    def extract_text(self, content: bytes, ext: str) -> str:
        """Route to the correct extractor based on file extension."""
        ext = ext.lower().lstrip('.')
        
        if ext == 'pdf':
            return self._extract_pdf(content)
        elif ext in ('docx', 'doc'):
            return self._extract_docx(content)
        elif ext == 'txt':
            return content.decode('utf-8', errors='replace')
        else:
            # Try plain text as fallback
            try:
                return content.decode('utf-8', errors='replace')
            except Exception:
                raise ValueError(f"Unsupported file format: .{ext}")

    def _extract_pdf(self, content: bytes) -> str:
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return self._clean('\n'.join(pages))
        except ImportError:
            raise RuntimeError(
                "pypdf is not installed. Install it with: pip install pypdf"
            )
        except Exception as e:
            raise RuntimeError(f"Could not parse PDF: {e}")

    def _extract_docx(self, content: bytes) -> str:
        try:
            import docx
            doc = docx.Document(io.BytesIO(content))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return self._clean('\n'.join(paragraphs))
        except ImportError:
            raise RuntimeError(
                "python-docx is not installed. Install it with: pip install python-docx"
            )
        except Exception as e:
            raise RuntimeError(f"Could not parse DOCX: {e}")

    def _clean(self, text: str) -> str:
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

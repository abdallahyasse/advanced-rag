from app.agent.models import ToolResult
from app.agent.tools.base import BaseTool
from app.config.settings import settings
from pypdf import PdfReader


class PDFTool(BaseTool):
    """
    PDF Inspection Tool.
    """

    name = "pdf"

    description = (
        "Inspect the indexed PDF document. "
        "Can return the total number of pages or PDF metadata."
    )

    parameters = {
        "type": "object",
        "properties": {
            "page": {
                "type": "integer",
                "description": (
                    "Optional page number to inspect. "
                    "If omitted, returns general PDF information."
                ),
            }
        },
        "required": [],
    }

    def run(
        self,
        page: int | None = None,
    ) -> ToolResult:

        try:

            reader = PdfReader(settings.pdf_path)

            result = {
                "pages": len(reader.pages),
                "metadata": (
                    dict(reader.metadata)
                    if reader.metadata
                    else {}
                ),
            }

            if page is not None:

                if page < 1 or page > len(reader.pages):

                    return ToolResult(
                        success=False,
                        tool=self.name,
                        error=(
                            f"Invalid page number: {page}. "
                            f"PDF contains {len(reader.pages)} pages."
                        ),
                    )

                page_text = reader.pages[page - 1].extract_text()

                result["page"] = page
                result["content"] = page_text or ""

            return ToolResult(
                success=True,
                tool=self.name,
                result=result,
            )

        except Exception as e:

            return ToolResult(
                success=False,
                tool=self.name,
                error=str(e),
            )
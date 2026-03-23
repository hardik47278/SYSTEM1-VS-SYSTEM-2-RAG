import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # ✅ changed

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LLM = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",  # ✅ changed from llama-3.1-8b-instant
    temperature=0,
    streaming=False,
)


class RichMetadata(BaseModel):
    title: str
    author: str
    summary: str
    keywords: List[str]
    year: int | None = None


METADATA_PROMPT = PromptTemplate(
    input_variables=["document_text", "format_instructions"],
    template="""
Extract structured metadata from the following document.
Return ONLY valid JSON (no code, no markdown, no backticks).

Fields:
- title
- author
- summary
- keywords
- year (if mentioned else null)

{format_instructions}

Document:
{document_text}
""".strip(),
)


def extract_rich_metadata(document_text: str) -> dict:
    parser = JsonOutputParser(pydantic_object=RichMetadata)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=LLM)

    chain = METADATA_PROMPT | LLM | fixing_parser

    result = chain.invoke(
        {
            "document_text": document_text[:8000],
            "format_instructions": parser.get_format_instructions(),
        }
    )

    if hasattr(result, "dict"):
        return result.dict()

    return result

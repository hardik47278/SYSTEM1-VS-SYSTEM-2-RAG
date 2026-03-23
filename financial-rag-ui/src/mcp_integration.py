
import asyncio
import json
import sys
import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent          
PROJECT_DIR = SRC_DIR.parent                         
RAG_SRC_DIR = Path(r"D:\ragg\RAG-Challenge-2\src")  
RAG_ROOT_DIR = Path(r"D:\ragg\RAG-Challenge-2")    

for p in [str(PROJECT_DIR), str(SRC_DIR), str(RAG_SRC_DIR), str(RAG_ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types


from upload import (
    create_file_index,
    answer_from_uploaded_file,
    stream_answer_from_uploaded_file,
    extract_text,
    generate_audio,
)
from richetadataextraction import extract_rich_metadata


_offline_processor = None

def get_offline_processor():
    """Lazy load QuestionsProcessor — only init when first needed."""
    global _offline_processor
    if _offline_processor is None:
        from pyprojroot import here
        from pipeline import Pipeline, max_nst_o3m_config
        from question_processing_copy import QuestionsProcessor

        root_path = Path(r"D:\ragg\RAG-Challenge-2\data\test_set")
        pipeline = Pipeline(root_path, run_config=max_nst_o3m_config)

        _offline_processor = QuestionsProcessor(
            vector_db_dir=pipeline.paths.vector_db_dir,
            documents_dir=pipeline.paths.documents_dir,
            subset_path=pipeline.paths.subset_path,
            new_challenge_pipeline=True,
            parent_document_retrieval=True,
            llm_reranking=True,
            answering_model="o3-mini-2025-01-31",
        )
    return _offline_processor


app = Server("rag-upload-server")



@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [

       
        types.Tool(
            name="ingest_file",
            description=(
                "Ingest a file (PDF, DOCX, PPTX, CSV, XLSX) into the RAG vector store. "
                "Returns a file_index ID to use for querying."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the file on disk"},
                    "filename": {"type": "string", "description": "Original filename including extension (e.g. report.pdf)"},
                },
                "required": ["file_path", "filename"],
            },
        ),

        types.Tool(
            name="query_file",
            description=(
                "Ask a question about an ingested file using RAG. "
                "Returns the answer, source chunks, and context precision score."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to answer from the document"},
                    "file_index": {"type": "string", "description": "The file_index ID returned by ingest_file"},
                },
                "required": ["question", "file_index"],
            },
        ),

        types.Tool(
            name="stream_query_file",
            description=(
                "Stream an answer from an ingested file. "
                "Returns a list of events (retrieved, answer chunks, done)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "file_index": {"type": "string"},
                },
                "required": ["question", "file_index"],
            },
        ),

        # ── OFFLINE PIPELINE TOOLS ─────────────────────────────
        types.Tool(
            name="check_company_match",
            description=(
                "Check if a question mentions any company that exists in the offline "
                "financial reports database (subset.csv). "
                "Returns list of matched company names. "
                "Use this BEFORE offline_query to decide routing."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to check for company mentions"},
                },
                "required": ["question"],
            },
        ),

        types.Tool(
            name="offline_query",
            description=(
                "Answer a financial question using the offline RAG pipeline "
                "(pre-indexed annual reports with vector DB + LLM reranking). "
                "Best for questions about specific companies in the database. "
                "Returns answer, references, and context precision score."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Financial question mentioning a specific company name",
                    },
                },
                "required": ["question"],
            },
        ),

        types.Tool(
            name="online_query",
            description=(
                "Answer a general financial question using OpenAI directly (no RAG). "
                "Use this as a fallback when no company match is found and no file is uploaded. "
                "Returns a best-effort answer."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The financial question to answer"},
                    "model": {
                        "type": "string",
                        "description": "OpenAI model to use (default: gpt-4o-mini-2024-07-18)",
                        "default": "gpt-4o-mini-2024-07-18",
                    },
                },
                "required": ["question"],
            },
        ),

        types.Tool(
            name="auto_route_query",
            description=(
                "Automatically route a financial question to the best pipeline: "
                "1) OFFLINE if company found in database, "
                "2) UPLOAD RAG if a file_index is provided, "
                "3) ONLINE as fallback. "
                "This is the recommended tool for general financial Q&A."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The financial question"},
                    "file_index": {
                        "type": "string",
                        "description": "Optional file_index from ingest_file for upload RAG route",
                    },
                },
                "required": ["question"],
            },
        ),

        # ── VOICE TOOLS ────────────────────────────────────────
        types.Tool(
            name="transcribe_audio",
            description=(
                "Transcribe a WAV audio file to text using OpenAI Whisper. "
                "Provide the absolute path to a .wav file on disk."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "audio_path": {"type": "string", "description": "Absolute path to the WAV audio file on disk"},
                },
                "required": ["audio_path"],
            },
        ),

        types.Tool(
            name="text_to_speech",
            description=(
                "Convert text to speech using ElevenLabs and save the audio file to disk. "
                "Returns the path to the saved audio file."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to convert to speech"},
                    "output_path": {"type": "string", "description": "Absolute path where the audio file will be saved (e.g. D:/ragg/output.mp3)"},
                    "voice_id": {"type": "string", "description": "ElevenLabs voice ID (optional, defaults to Rachel voice)"},
                },
                "required": ["text", "output_path"],
            },
        ),

        # ── METADATA TOOL ──────────────────────────────────────
        types.Tool(
            name="extract_metadata",
            description=(
                "Extract rich metadata (title, author, summary, keywords, year) "
                "from a block of text. Useful for understanding document content quickly."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to extract metadata from"},
                },
                "required": ["text"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

   
    if name == "ingest_file":
        try:
            file_index = await asyncio.to_thread(
                create_file_index, arguments["file_path"], arguments["filename"]
            )
            return [types.TextContent(type="text", text=json.dumps({"file_index": file_index, "status": "success"}))]
        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"error": str(e), "status": "failed"}))]

    
    elif name == "query_file":
        try:
            result = await asyncio.to_thread(
                answer_from_uploaded_file, arguments["question"], arguments["file_index"]
            )
            return [types.TextContent(type="text", text=json.dumps(result, default=str))]
        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]

  
    elif name == "stream_query_file":
        try:
            events = []
            for event in stream_answer_from_uploaded_file(arguments["question"], arguments["file_index"]):
                events.append(event)
            return [types.TextContent(type="text", text=json.dumps(events, default=str))]
        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


    elif name == "check_company_match":
        try:
            processor = await asyncio.to_thread(get_offline_processor)
            companies = await asyncio.to_thread(
                processor._extract_companies_from_subset, arguments["question"]
            )
            return [types.TextContent(type="text", text=json.dumps({
                "matched_companies": companies,
                "match_found": len(companies) > 0,
                "status": "success"
            }))]
        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"error": str(e), "status": "failed"}))]

  
    elif name == "offline_query":
        try:
            processor = await asyncio.to_thread(get_offline_processor)
            result = await asyncio.to_thread(processor.process_question, arguments["question"])
            return [types.TextContent(type="text", text=json.dumps(result, default=str))]
        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"error": str(e), "status": "failed"}))]

  
    elif name == "online_query":
        try:
            processor = await asyncio.to_thread(get_offline_processor)
            model = arguments.get("model", "gpt-4o-mini-2024-07-18")
            result = await asyncio.to_thread(
                processor.openai_processor.get_answer_online,
                arguments["question"],
                model
            )
            return [types.TextContent(type="text", text=json.dumps(result, default=str))]
        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"error": str(e), "status": "failed"}))]

    
    elif name == "auto_route_query":
        question = arguments["question"]
        file_index = arguments.get("file_index")

        try:
            processor = await asyncio.to_thread(get_offline_processor)

           
            companies = await asyncio.to_thread(
                processor._extract_companies_from_subset, question
            )
            if companies:
                result = await asyncio.to_thread(processor.process_question, question)
                return [types.TextContent(type="text", text=json.dumps({
                    **result,
                    "route": "OFFLINE",
                    "matched_companies": companies,
                    "status": "success"
                }, default=str))]

            
            if file_index:
                result = await asyncio.to_thread(
                    answer_from_uploaded_file, question, file_index
                )
                return [types.TextContent(type="text", text=json.dumps({
                    **result,
                    "route": "UPLOAD_RAG",
                    "status": "success"
                }, default=str))]

            
            model = "gpt-4o-mini-2024-07-18"
            result = await asyncio.to_thread(
                processor.openai_processor.get_answer_online, question, model
            )
            return [types.TextContent(type="text", text=json.dumps({
                **result,
                "route": "ONLINE",
                "status": "success"
            }, default=str))]

        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"error": str(e), "status": "failed"}))]

    
    elif name == "transcribe_audio":
        try:
            from io import BytesIO
            with open(arguments["audio_path"], "rb") as f:
                audio_buffer = BytesIO(f.read())
            transcription = await asyncio.to_thread(extract_text, audio_buffer)
            return [types.TextContent(type="text", text=json.dumps({"transcription": transcription, "status": "success"}))]
        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"error": str(e), "status": "failed"}))]

   
    elif name == "text_to_speech":
        try:
            voice_id = arguments.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
            audio_buffer = await asyncio.to_thread(generate_audio, arguments["text"], "en", voice_id)
            output = Path(arguments["output_path"])
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "wb") as f:
                f.write(audio_buffer.read())
            return [types.TextContent(type="text", text=json.dumps({
                "output_path": str(output),
                "status": "success",
                "message": f"Audio saved to {arguments['output_path']}"
            }))]
        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"error": str(e), "status": "failed"}))]

   
    elif name == "extract_metadata":
        try:
            metadata = await asyncio.to_thread(extract_rich_metadata, arguments["text"])
            return [types.TextContent(type="text", text=json.dumps({"metadata": metadata, "status": "success"}))]
        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"error": str(e), "status": "failed"}))]

    else:
        return [types.TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )

if __name__ == "__main__":
    asyncio.run(main())

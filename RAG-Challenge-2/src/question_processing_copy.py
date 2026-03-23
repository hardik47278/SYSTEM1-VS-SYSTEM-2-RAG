import json
from typing import Union, Dict, List, Optional
import re
from pathlib import Path
from src.retrieval import VectorRetriever, HybridRetriever
from src.api_req import APIProcessor
from tqdm import tqdm
import pandas as pd
import threading
import concurrent.futures
import hashlib
from datetime import datetime, timedelta
import logging

from ragas.llms import llm_factory
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference
import os
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE: Dict[str, Dict] = {}
CACHE_TTL = timedelta(hours=1)


class QuestionsProcessor:
    def __init__(
        self,
        vector_db_dir: Union[str, Path] = "./vector_dbs",
        documents_dir: Union[str, Path] = "./documents",
        questions_file_path: Optional[Union[str, Path]] = None,
        new_challenge_pipeline: bool = False,
        subset_path: Optional[Union[str, Path]] = None,
        parent_document_retrieval: bool = False,
        llm_reranking: bool = False,
        llm_reranking_sample_size: int = 20,
        top_n_retrieval: int = 10,
        parallel_requests: int = 10,
        api_provider: str = "openai",
        answering_model: str = "gpt-4o-2024-08-06",
        full_context: bool = False,
    ):
        self.questions = self._load_questions(questions_file_path)
        self.documents_dir = Path(documents_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.subset_path = Path(subset_path) if subset_path else None

        self.new_challenge_pipeline = new_challenge_pipeline
        self.return_parent_pages = parent_document_retrieval
        self.llm_reranking = llm_reranking
        self.llm_reranking_sample_size = llm_reranking_sample_size
        self.top_n_retrieval = top_n_retrieval
        self.answering_model = answering_model
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.openai_processor = APIProcessor(provider=api_provider)
        self.full_context = full_context

        self.answer_details: List[Optional[dict]] = []
        self.detail_counter = 0
        self._lock = threading.Lock()
        self.enable_ragas = True

        # Load .env so OPENAI_API_KEY is available for RAGAS evaluator
        load_dotenv()

        # used by subset extraction / references
        self.companies_df = None

        # stored per question call
        self.response_data = None

        # -------------------------
        # RAGAS evaluator LLM
        # -------------------------
        self.evaluator_llm = None
        if self.enable_ragas:
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found. Check .env location and load_dotenv().")

                client = OpenAI(api_key=api_key)

                # Use cheap evaluator model (answering model stays unchanged)
                self.evaluator_llm = llm_factory("gpt-4o-mini", client=client)
            except Exception as e:
                logger.warning(f"Failed to init RAGAS evaluator LLM: {e}")
                self.enable_ragas = False

    def _load_questions(self, questions_file_path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
        if questions_file_path is None:
            return []
        with open(questions_file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _format_retrieval_results(self, retrieval_results) -> str:
        """Format vector retrieval results into RAG context string"""
        if not retrieval_results:
            return ""

        context_parts = []
        for result in retrieval_results:
            page_number = result["page"]
            text = result["text"]
            context_parts.append(f'Text retrieved from page {page_number}: \n"""\n{text}\n"""')

        return "\n\n---\n\n".join(context_parts)

    def _make_cache_key(self, company_name: str, question: str) -> str:
        raw = f"{company_name}|{question}|{self.answering_model}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str):
        item = CACHE.get(key)
        if not item:
            logger.info("🐢 OFFLINE CACHE MISS")
            print("🐢 CACHE MISS")
            return None

        if datetime.now() - item["created_at"] > CACHE_TTL:
            del CACHE[key]
            logger.info("⌛ OFFLINE CACHE EXPIRED")
            return None

        print("⚡ CACHE HIT")
        logger.info("⚡ OFFLINE CACHE HIT")
        return item["answer"]

    def _cache_set(self, key: str, answer: dict):
        CACHE[key] = {"answer": answer, "created_at": datetime.now()}

    def _extract_references(self, pages_list: list, company_name: str) -> list:
        if self.subset_path is None:
            raise ValueError("subset_path is required for new challenge pipeline when processing references.")

        # Load only once if possible
        if self.companies_df is None:
            self.companies_df = pd.read_csv(self.subset_path)

        matching_rows = self.companies_df[self.companies_df["company_name"] == company_name]
        if matching_rows.empty:
            company_sha1 = ""
        else:
            company_sha1 = matching_rows.iloc[0]["sha1"]

        refs = [{"pdf_sha1": company_sha1, "page_index": page} for page in pages_list]
        return refs

    def _validate_page_references(
        self,
        claimed_pages: list,
        retrieval_results: list,
        min_pages: int = 2,
        max_pages: int = 8,
    ) -> list:
        """
        Validate that all page numbers mentioned in the LLM's answer are actually from the retrieval results.
        If fewer than min_pages valid references remain, add top pages from retrieval results.
        """
        if claimed_pages is None:
            claimed_pages = []

        retrieved_pages = [result["page"] for result in retrieval_results] if retrieval_results else []
        validated_pages = [page for page in claimed_pages if page in retrieved_pages]

        if len(validated_pages) < len(claimed_pages):
            removed_pages = set(claimed_pages) - set(validated_pages)
            print(f"Warning: Removed {len(removed_pages)} hallucinated page references: {removed_pages}")

        if len(validated_pages) < min_pages and retrieval_results:
            existing_pages = set(validated_pages)
            for result in retrieval_results:
                page = result["page"]
                if page not in existing_pages:
                    validated_pages.append(page)
                    existing_pages.add(page)
                    if len(validated_pages) >= min_pages:
                        break

        if len(validated_pages) > max_pages:
            print(f"Trimming references from {len(validated_pages)} to {max_pages} pages")
            validated_pages = validated_pages[:max_pages]

        return validated_pages

    def _compute_context_precison_sync(self, question: str, final_answer: str, context: str) -> float:
        if not self.evaluator_llm:
            raise RuntimeError("RAGAS evaluator LLM is not initialized")

        metric = LLMContextPrecisionWithoutReference(llm=self.evaluator_llm)
        sample = SingleTurnSample(
            user_input=question,
            response=final_answer,
            retrieved_contexts=[context],
        )

        score = metric.single_turn_score(sample)
        logger.info("context precison score: {score}")
        return float(score)

    def get_answer_for_company(self, company_name: str, question: str) -> dict:
        """
        Full logic (NOT SKIPPED):
        1) cache lookup
        2) choose retriever (HybridRetriever if llm_reranking else VectorRetriever)
        3) retrieval (retrieve_all if full_context else retrieve_by_company_name)
        4) format context
        5) LLM answer
        6) validate refs (new challenge)
        7) cache set + return
        """

        cache_key = self._make_cache_key(company_name, question)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # ✅ CRITICAL FIX: always define retrieval_results (prevents UnboundLocalError)
        retrieval_results = []

        # choose retriever
        if self.llm_reranking:
            retriever = HybridRetriever(vector_db_dir=self.vector_db_dir, documents_dir=self.documents_dir)
        else:
            retriever = VectorRetriever(vector_db_dir=self.vector_db_dir, documents_dir=self.documents_dir)

        # retrieval
        if self.full_context:
            retrieval_results = retriever.retrieve_all(company_name)
        else:
            retrieval_results = retriever.retrieve_by_company_name(
                company_name=company_name,
                query=question,
                llm_reranking_sample_size=self.llm_reranking_sample_size,
                top_n=self.top_n_retrieval,
                return_parent_pages=self.return_parent_pages,
            )

        if not retrieval_results:
            raise ValueError("No relevant context found")

        rag_context = self._format_retrieval_results(retrieval_results)

        answer_dict = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            model=self.answering_model,
        )

        self.response_data = self.openai_processor.response_data

        if getattr(self, "enable_ragas", False):
            try:
                score = self._compute_context_precison_sync(
                    question=question,
                    final_answer=answer_dict.get("final_answer", ""),
                    context=rag_context,
                )
                answer_dict["context_Precison"] = score
                print("Context Precision Score:", score)

            except Exception as e:
                logger.warning(f"RAGAS context precision failed: {e}")

        if self.new_challenge_pipeline:
            pages = answer_dict.get("relevant_pages", [])
            validated_pages = self._validate_page_references(pages, retrieval_results)
            answer_dict["relevant_pages"] = validated_pages
            answer_dict["references"] = self._extract_references(validated_pages, company_name)

        self._cache_set(cache_key, answer_dict)
        return answer_dict

    def _extract_companies_from_subset(self, question_text: str) -> list[str]:
        """Extract company names from a question by matching against companies in the subset file."""
        if self.subset_path is None:
            raise ValueError("subset_path must be provided to use subset extraction")

        if self.companies_df is None:
            self.companies_df = pd.read_csv(self.subset_path)

        found_companies = []
        company_names = sorted(self.companies_df["company_name"].unique(), key=len, reverse=True)

        for company in company_names:
            escaped_company = re.escape(company)
            pattern = rf"{escaped_company}(?:\W|$)"
            if re.search(pattern, question_text, re.IGNORECASE):
                found_companies.append(company)
                question_text = re.sub(pattern, "", question_text, flags=re.IGNORECASE)

        return found_companies

    def process_question(self, question: str):
        if self.new_challenge_pipeline:
            extracted_companies = self._extract_companies_from_subset(question)
        else:
            extracted_companies = re.findall(r'"([^"]*)"', question)

        if len(extracted_companies) == 0:
            raise ValueError("No company name found in the question.")

        if len(extracted_companies) == 1:
            company_name = extracted_companies[0]
            answer_dict = self.get_answer_for_company(company_name=company_name, question=question)
            return answer_dict

        return self.process_comparative_question(question, extracted_companies)

    def _create_answer_detail_ref(self, answer_dict: dict, question_index: int) -> str:
        """Create a reference ID for answer details and store the details"""
        ref_id = f"#/answer_details/{question_index}"
        with self._lock:
            self.answer_details[question_index] = {
                "step_by_step_analysis": answer_dict.get("step_by_step_analysis"),
                "reasoning_summary": answer_dict.get("reasoning_summary"),
                "relevant_pages": answer_dict.get("relevant_pages"),
                "response_data": self.response_data,
                "self": ref_id,
            }
        return ref_id

    def _calculate_statistics(self, processed_questions: List[dict], print_stats: bool = False) -> dict:
        total_questions = len(processed_questions)
        error_count = sum(1 for q in processed_questions if "error" in q)
        na_count = sum(
            1
            for q in processed_questions
            if (q.get("value") if "value" in q else q.get("answer")) == "N/A"
        )
        success_count = total_questions - error_count - na_count

        if print_stats:
            print("\nFinal Processing Statistics:")
            print(f"Total questions: {total_questions}")
            print(f"Errors: {error_count} ({(error_count / total_questions) * 100:.1f}%)")
            print(f"N/A answers: {na_count} ({(na_count / total_questions) * 100:.1f}%)")
            print(f"Successfully answered: {success_count} ({(success_count / total_questions) * 100:.1f}%)\n")

        return {
            "total_questions": total_questions,
            "error_count": error_count,
            "na_count": na_count,
            "success_count": success_count,
        }

    def process_questions_list(
        self,
        questions_list: List[dict],
        output_path: str = None,
        submission_file: bool = False,
        team_email: str = "",
        submission_name: str = "",
        pipeline_details: str = "",
    ) -> dict:
        total_questions = len(questions_list)
        questions_with_index = [{**q, "_question_index": i} for i, q in enumerate(questions_list)]

        self.answer_details = [None] * total_questions
        processed_questions = []
        parallel_threads = self.parallel_requests

        if parallel_threads <= 1:
            for question_data in tqdm(questions_with_index, desc="Processing questions"):
                processed_question = self._process_single_question(question_data)
                processed_questions.append(processed_question)
                if output_path:
                    self._save_progress(
                        processed_questions,
                        output_path,
                        submission_file=submission_file,
                        team_email=team_email,
                        submission_name=submission_name,
                        pipeline_details=pipeline_details,
                    )
        else:
            with tqdm(total=total_questions, desc="Processing questions") as pbar:
                for i in range(0, total_questions, parallel_threads):
                    batch = questions_with_index[i: i + parallel_threads]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                        batch_results = list(executor.map(self._process_single_question, batch))
                    processed_questions.extend(batch_results)

                    if output_path:
                        self._save_progress(
                            processed_questions,
                            output_path,
                            submission_file=submission_file,
                            team_email=team_email,
                            submission_name=submission_name,
                            pipeline_details=pipeline_details,
                        )
                    pbar.update(len(batch_results))

        statistics = self._calculate_statistics(processed_questions, print_stats=True)
        return {"questions": processed_questions, "answer_details": self.answer_details, "statistics": statistics}

    def _process_single_question(self, question_data: dict) -> dict:
        question_index = question_data.get("_question_index", 0)

        if self.new_challenge_pipeline:
            question_text = question_data.get("text")
        else:
            question_text = question_data.get("question")

        try:
            answer_dict = self.process_question(question_text)

            if "error" in answer_dict:
                detail_ref = self._create_answer_detail_ref(
                    {"step_by_step_analysis": None, "reasoning_summary": None, "relevant_pages": None},
                    question_index,
                )
                if self.new_challenge_pipeline:
                    return {
                        "question_text": question_text,
                        "value": None,
                        "references": [],
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref},
                    }
                return {
                    "question": question_text,
                    "answer": None,
                    "error": answer_dict["error"],
                    "answer_details": {"$ref": detail_ref},
                }

            detail_ref = self._create_answer_detail_ref(answer_dict, question_index)

            if self.new_challenge_pipeline:
                return {
                    "question_text": question_text,
                    "value": answer_dict.get("final_answer"),
                    "references": answer_dict.get("references", []),
                    "answer_details": {"$ref": detail_ref},
                }

            return {
                "question": question_text,
                "answer": answer_dict.get("final_answer"),
                "answer_details": {"$ref": detail_ref},
            }

        except Exception as err:
            return self._handle_processing_error(question_text, err, question_index)

    def _handle_processing_error(self, question_text: str, err: Exception, question_index: int) -> dict:
        import traceback

        error_message = str(err)
        tb = traceback.format_exc()
        error_ref = f"#/answer_details/{question_index}"
        error_detail = {"error_traceback": tb, "self": error_ref}

        with self._lock:
            self.answer_details[question_index] = error_detail

        print(f"Error encountered processing question: {question_text}")
        print(f"Error type: {type(err).__name__}")
        print(f"Error message: {error_message}")
        print(f"Full traceback:\n{tb}\n")

        if self.new_challenge_pipeline:
            return {
                "question_text": question_text,
                "value": None,
                "references": [],
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref},
            }

        return {
            "question": question_text,
            "answer": None,
            "error": f"{type(err).__name__}: {error_message}",
            "answer_details": {"$ref": error_ref},
        }

    def _post_process_submission_answers(self, processed_questions: List[dict]) -> List[dict]:
        submission_answers = []

        for q in processed_questions:
            question_text = q.get("question_text") or q.get("question")
            value = "N/A" if "error" in q else (q.get("value") if "value" in q else q.get("answer"))
            references = q.get("references", [])

            answer_details_ref = q.get("answer_details", {}).get("$ref", "")
            step_by_step_analysis = None
            if answer_details_ref and answer_details_ref.startswith("#/answer_details/"):
                try:
                    index = int(answer_details_ref.split("/")[-1])
                    if 0 <= index < len(self.answer_details) and self.answer_details[index]:
                        step_by_step_analysis = self.answer_details[index].get("step_by_step_analysis")
                except (ValueError, IndexError):
                    pass

            if value == "N/A":
                references = []
            else:
                references = [{"pdf_sha1": ref["pdf_sha1"], "page_index": ref["page_index"] - 1} for ref in references]

            submission_answer = {"question_text": question_text, "value": value, "references": references}
            if step_by_step_analysis:
                submission_answer["reasoning_process"] = step_by_step_analysis

            submission_answers.append(submission_answer)

        return submission_answers

    def _save_progress(
        self,
        processed_questions: List[dict],
        output_path: Optional[str],
        submission_file: bool = False,
        team_email: str = "",
        submission_name: str = "",
        pipeline_details: str = "",
    ):
        if not output_path:
            return

        statistics = self._calculate_statistics(processed_questions)

        result = {"questions": processed_questions, "answer_details": self.answer_details, "statistics": statistics}

        output_file = Path(output_path)
        debug_file = output_file.with_name(output_file.stem + "_debug" + output_file.suffix)

        with open(debug_file, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

        if submission_file:
            submission_answers = self._post_process_submission_answers(processed_questions)
            submission = {
                "answers": submission_answers,
                "team_email": team_email,
                "submission_name": submission_name,
                "details": pipeline_details,
            }
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(submission, file, ensure_ascii=False, indent=2)

    def process_all_questions(
        self,
        output_path: str = "questions_with_answers.json",
        team_email: str = "79250515615@yandex.com",
        submission_name: str = "Ilia_Ris SO CoT + Parent Document Retrieval",
        submission_file: bool = False,
        pipeline_details: str = "",
    ):
        return self.process_questions_list(
            self.questions,
            output_path,
            submission_file=submission_file,
            team_email=team_email,
            submission_name=submission_name,
            pipeline_details=pipeline_details,
        )

    def process_comparative_question(self, question: str, companies: List[str]) -> dict:
        rephrased_questions = self.openai_processor.get_rephrased_questions(
            original_question=question, companies=companies
        )

        individual_answers = {}
        aggregated_references = []

        def process_company_question(company: str) -> tuple[str, dict]:
            sub_question = rephrased_questions.get(company)
            if not sub_question:
                raise ValueError(f"Could not generate sub-question for company: {company}")

            answer_dict = self.get_answer_for_company(company_name=company, question=sub_question)
            return company, answer_dict

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_company = {executor.submit(process_company_question, company): company for company in companies}

            for future in concurrent.futures.as_completed(future_to_company):
                company, answer_dict = future.result()
                individual_answers[company] = answer_dict
                aggregated_references.extend(answer_dict.get("references", []))

        unique_refs = {}
        for ref in aggregated_references:
            key = (ref.get("pdf_sha1"), ref.get("page_index"))
            unique_refs[key] = ref
        aggregated_references = list(unique_refs.values())

        comparative_answer = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            model=self.answering_model,
        )

        self.response_data = self.openai_processor.response_data
        comparative_answer["references"] = aggregated_references
        return comparative_answer

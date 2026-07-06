import copy
import os
import re
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from edison_client import EdisonClient, JobNames
from lmi import LiteLLMModel
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from .prompts import (
    ANALYSIS_QUERIES,
    ASSAY_HYPOTHESIS_FORMAT,
    ASSAY_HYPOTHESIS_SYSTEM_PROMPT,
    ASSAY_LITERATURE_SYSTEM_MESSAGE,
    ASSAY_LITERATURE_USER_MESSAGE,
    ASSAY_PROPOSAL_SYSTEM_MESSAGE,
    ASSAY_PROPOSAL_USER_MESSAGE,
    ASSAY_RANKING_PROMPT_FORMAT,
    ASSAY_RANKING_SYSTEM_PROMPT,
    CANDIDATE_GENERATION_SYSTEM_MESSAGE,
    CANDIDATE_GENERATION_USER_MESSAGE,
    CANDIDATE_LIT_REVIEW_DIRECTION_PROMPT,
    CANDIDATE_QUERY_GENERATION_CONTENT_MESSAGE,
    CANDIDATE_QUERY_GENERATION_SYSTEM_MESSAGE,
    CANDIDATE_RANKING_PROMPT_FORMAT,
    CANDIDATE_RANKING_SYSTEM_PROMPT,
    CANDIDATE_REPORT_FORMAT,
    CHAIN_OF_THOUGHT_AGNOSTIC,
    CONSENSUS_QUERIES,
    COT,
    DATA_INTERPRETATION_CONTENT_MESSAGE,
    DATA_INTERPRETATION_SYSTEM_MESSAGE,
    EXPERIMENTAL_INSIGHTS_APPENDAGE,
    EXPERIMENTAL_INSIGHTS_FOR_CANDIDATE_GENERATION,
    FOLLOWUP_CONTENT_MESSAGE,
    FOLLOWUP_SYSTEM_MESSAGE,
    GENERAL_NOTEBOOK_GUIDELINES,
    GUIDELINE,
    R_SPECIFIC_GUIDELINES,
    SYNTHESIZE_SYSTEM_MESSAGE_CONTENT,
    SYNTHESIZE_USER_CONTENT,
)

load_dotenv()

_DEFAULT_LLM_CONFIG_DATA = {
    "model_list": [
        {
            "model_name": "o4-mini",
            "litellm_params": {
                "model": "o4-mini",
                "api_key": "",
                "timeout": 300,
            },
        }
    ]
}


def get_default_llm_name() -> str:
    return (
        os.getenv("ROBIN_LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or _DEFAULT_LLM_CONFIG_DATA["model_list"][0]["model_name"]
    )


def get_llm_api_key(model_name: str) -> str:
    if model_name.startswith("gemini/"):
        return os.getenv("GEMINI_API_KEY") or os.getenv(
            "GOOGLE_API_KEY", "insert_gemini_key_here"
        )
    if model_name.startswith("groq/"):
        return os.getenv("GROQ_API_KEY", "insert_groq_key_here")
    if model_name.startswith("anthropic/"):
        return os.getenv("ANTHROPIC_API_KEY", "insert_anthropic_key_here")
    return os.getenv("OPENAI_API_KEY", "insert_openai_key_here")


def get_default_llm_config() -> dict[str, Any]:
    # Env vars are read on each instantiation so values set after import are picked up.
    model_name = get_default_llm_name()
    data: dict[str, Any] = copy.deepcopy(_DEFAULT_LLM_CONFIG_DATA)
    data["model_list"][0]["model_name"] = model_name
    data["model_list"][0]["litellm_params"]["model"] = model_name
    data["model_list"][0]["litellm_params"]["api_key"] = get_llm_api_key(model_name)
    return data


def _get_prompt_args(template_string: str) -> set[str]:
    """
    Extracts root variable names from f-string like placeholders (e.g., {variable})
    using a direct regex approach.
    """  # noqa: D205
    placeholders: set[str] = set()
    placeholders.update(
        match.group(1)
        for match in re.finditer(
            r"(?<!{){([a-zA-Z_][a-zA-Z0-9_]*)[^}]*}(?!})", template_string
        )
    )
    return placeholders


class Prompts(BaseModel):
    analysis_queries: dict[str, str] = Field(default_factory=lambda: ANALYSIS_QUERIES)
    consensus_queries: dict[str, str] = Field(default_factory=lambda: CONSENSUS_QUERIES)
    assay_literature_system_message: str = Field(
        default=ASSAY_LITERATURE_SYSTEM_MESSAGE
    )
    assay_literature_user_message: str = Field(default=ASSAY_LITERATURE_USER_MESSAGE)
    assay_proposal_system_message: str = Field(default=ASSAY_PROPOSAL_SYSTEM_MESSAGE)
    assay_proposal_user_message: str = Field(default=ASSAY_PROPOSAL_USER_MESSAGE)
    assay_hypothesis_system_prompt: str = Field(default=ASSAY_HYPOTHESIS_SYSTEM_PROMPT)
    assay_hypothesis_format: str = Field(default=ASSAY_HYPOTHESIS_FORMAT)
    assay_ranking_system_prompt: str = Field(default=ASSAY_RANKING_SYSTEM_PROMPT)
    assay_ranking_prompt_format: str = Field(default=ASSAY_RANKING_PROMPT_FORMAT)
    synthesize_user_content: str = Field(default=SYNTHESIZE_USER_CONTENT)
    synthesize_system_message_content: str = Field(
        default=SYNTHESIZE_SYSTEM_MESSAGE_CONTENT
    )
    candidate_query_generation_system_message: str = Field(
        default=CANDIDATE_QUERY_GENERATION_SYSTEM_MESSAGE
    )
    experimental_insights_appendage: str = Field(
        default=EXPERIMENTAL_INSIGHTS_APPENDAGE
    )
    candidate_query_generation_content_message: str = Field(
        default=CANDIDATE_QUERY_GENERATION_CONTENT_MESSAGE
    )
    candidate_generation_system_message: str = Field(
        default=CANDIDATE_GENERATION_SYSTEM_MESSAGE
    )
    candidate_generation_user_message: str = Field(
        default=CANDIDATE_GENERATION_USER_MESSAGE
    )
    experimental_insights_for_candidate_generation: str = Field(
        default=EXPERIMENTAL_INSIGHTS_FOR_CANDIDATE_GENERATION
    )
    candidate_lit_review_direction_prompt: str = Field(
        default=CANDIDATE_LIT_REVIEW_DIRECTION_PROMPT
    )
    candidate_report_format: str = Field(default=CANDIDATE_REPORT_FORMAT)
    candidate_ranking_system_prompt: str = Field(
        default=CANDIDATE_RANKING_SYSTEM_PROMPT
    )
    candidate_ranking_prompt_format: str = Field(
        default=CANDIDATE_RANKING_PROMPT_FORMAT
    )
    cot: str = Field(default=COT)
    guideline: str = Field(default=GUIDELINE)
    data_interpretation_system_message: str = Field(
        default=DATA_INTERPRETATION_SYSTEM_MESSAGE
    )
    data_interpretation_content_message: str = Field(
        default=DATA_INTERPRETATION_CONTENT_MESSAGE
    )
    followup_system_message: str = Field(default=FOLLOWUP_SYSTEM_MESSAGE)
    followup_content_message: str = Field(default=FOLLOWUP_CONTENT_MESSAGE)
    general_notebook_guidelines: str = Field(default=GENERAL_NOTEBOOK_GUIDELINES)
    r_specific_guidelines: str = Field(default=R_SPECIFIC_GUIDELINES)
    cot_agnostic: str = Field(default=CHAIN_OF_THOUGHT_AGNOSTIC)

    @model_validator(mode="after")
    def validate_all_prompts(self) -> "Prompts":
        current_prompt_expectations: dict[str, set[str]] = {
            "data_interpretation_content_message": {"goal", "data_html"},
            "followup_content_message": {
                "goal",
                "analysis_summary",
                "mechanistic_insights",
                "questions_raised",
            },
            "assay_literature_system_message": {"num_assays"},
            "assay_literature_user_message": {"num_queries", "disease_name"},
            "assay_proposal_system_message": {"num_assays"},
            "assay_proposal_user_message": {
                "num_assays",
                "disease_name",
                "assay_lit_review_output",
            },
            "assay_hypothesis_system_prompt": {"disease_name"},
            "assay_hypothesis_format": {"disease_name"},
            "assay_ranking_system_prompt": {"disease_name"},
            "synthesize_user_content": {"assay_name", "disease_name"},
            "synthesize_system_message_content": {"disease_name"},
            "candidate_query_generation_system_message": {"disease_name"},
            "experimental_insights_appendage": {
                "candidate_generation_goal",
                "experimental_insights_analysis_summary",
                "experimental_insights_mechanistic_insights",
                "experimental_insights_questions_raised",
            },
            "candidate_query_generation_content_message": {
                "num_queries",
                "double_queries",
                "candidate_generation_goal",
                "disease_name",
            },
            "candidate_generation_system_message": {"disease_name", "num_candidates"},
            "candidate_generation_user_message": {
                "num_candidates",
                "disease_name",
                "therapeutic_candidate_review_output",
            },
            "experimental_insights_for_candidate_generation": {
                "candidate_generation_goal",
                "experimental_insights_analysis_summary",
                "experimental_insights_mechanistic_insights",
                "experimental_insights_questions_raised",
            },
            "candidate_lit_review_direction_prompt": {"disease_name"},
            "candidate_report_format": {"disease_name"},
            "candidate_ranking_system_prompt": {"disease_name"},
            "cot": set(),
            "guideline": set(),
            "assay_ranking_prompt_format": set(),
            "candidate_ranking_prompt_format": set(),
            "data_interpretation_system_message": set(),
            "followup_system_message": set(),
            "analysis_queries": set(),
            "consensus_queries": set(),
        }

        for field_name, expected_args in current_prompt_expectations.items():
            if not hasattr(self, field_name):
                raise ValueError(
                    f"Prompt field '{field_name}' defined in PROMPT_EXPECTATIONS but"
                    " not found in Prompts model."
                )

            prompt_template_value = getattr(self, field_name)

            if isinstance(prompt_template_value, dict):
                continue

            if not isinstance(prompt_template_value, str):
                raise TypeError(f"Prompt field '{field_name}' is not a string type.")

            actual_placeholders = _get_prompt_args(prompt_template_value)

            missing_in_template = expected_args - actual_placeholders
            if missing_in_template:
                raise ValueError(
                    f"Prompt '{field_name}' is missing expected placeholders:"
                    f" {missing_in_template}. Expected: {sorted(expected_args)}, Found:"
                    f" {sorted(actual_placeholders)}"
                )

            unexpected_in_template = actual_placeholders - expected_args
            if unexpected_in_template:
                raise ValueError(
                    f"Prompt '{field_name}' contains unexpected placeholders:"
                    f" {unexpected_in_template}. Expected: {sorted(expected_args)},"
                    f" Found: {sorted(actual_placeholders)}"
                )

        return self


def build_information_engineering_prompts() -> Prompts:
    prompts = Prompts()
    prompts.assay_literature_system_message = (
        "You are a senior information engineering research strategist. Your task is "
        "to identify rigorous study designs, evaluation protocols, benchmark setups, "
        "and empirical research hypotheses for a computing research topic. Generate "
        "exactly {num_assays} distinct ideas."
    )
    prompts.assay_literature_user_message = (
        "Return a list of {num_queries} queries (separated by <>) that would be useful "
        "for designing rigorous empirical studies, benchmarks, evaluation protocols, "
        "or research hypotheses for the information engineering topic: {disease_name}. "
        "The queries should cover algorithms, system designs, datasets, evaluation metrics, "
        "baselines, reproducibility issues, user or deployment contexts, and open research "
        "gaps. Each query should be 30+ words and suitable for searching scholarly literature."
    )
    prompts.assay_proposal_system_message = (
        "You are a meticulous information engineering researcher. Your task is to propose "
        "high-quality empirical study designs or evaluation protocols. Generate exactly "
        "{num_assays} distinct ideas. Focus on feasibility, methodological rigor, measurable "
        "outcomes, novelty, reproducibility, and relevance to real computing research practice.\n\n"
        "Output Format Specification (Strict Adherence Required):\n\n"
        "Your entire output MUST be a single, valid JSON array containing exactly "
        "{num_assays} objects. Each object MUST have this structure:\n\n"
        "[\n"
        "  {{\n"
        "    \"strategy_name\": \"string\",\n"
        "    \"reasoning\": \"string\"\n"
        "  }}\n"
        "]"
    )
    prompts.assay_proposal_user_message = (
        "Generate exactly {num_assays} distinct and rigorous proposals for empirical "
        "study designs, benchmark experiments, or evaluation protocols for the information "
        "engineering topic {disease_name}. Here is relevant background literature:\n"
        "{assay_lit_review_output}\n"
    )
    prompts.assay_hypothesis_system_prompt = (
        "You are an information engineering research lead evaluating a proposed empirical "
        "study or evaluation protocol for {disease_name}. Given the following proposal, "
        "perform a detailed literature-grounded evaluation of whether the study design is "
        "useful, feasible, reproducible, and capable of producing meaningful research insight."
    )
    prompts.assay_hypothesis_format = (
        "Provide your response in the following format, like an evaluation for a research proposal:\n"
        "Study Design Overview: Explain the proposed study, benchmark, protocol, datasets, "
        "systems, algorithms, measurements, and expected outputs.\n"
        "Research Evidence: Summarize literature showing why this topic, problem setting, "
        "or evaluation target matters in information engineering.\n"
        "Previous Use: Explain whether similar benchmarks, datasets, methods, or evaluation "
        "protocols have been used before and what they revealed.\n"
        "Overall Evaluation: Discuss strengths, weaknesses, feasibility, reproducibility, "
        "and likely insight value for {disease_name}."
    )
    prompts.assay_ranking_system_prompt = (
        "You are an experienced information engineering program committee member. Your "
        "objective is to compare two empirical study or evaluation protocol proposals for "
        "{disease_name}. Evaluate strictly on methodological rigor, feasibility, novelty, "
        "reproducibility, relevance of datasets/metrics/baselines, and potential research impact. "
        "Choose the proposal most likely to produce reliable and useful research insight."
    )
    prompts.synthesize_system_message_content = (
        "You are an information engineering researcher turning an evaluation protocol into "
        "a concise next-stage research goal for {disease_name}."
    )
    prompts.synthesize_user_content = (
        "Here is a proposed study or evaluation protocol for the topic '{disease_name}':\n\n"
        "Protocol Name: \"{assay_name}\"\n"
        "Synthesize a concise and specific research goal for the next stage, focused on "
        "identifying promising methods, algorithms, system designs, datasets, or benchmark "
        "directions to investigate for {disease_name}. Provide ONLY the synthesized goal string."
    )
    prompts.candidate_query_generation_system_message = (
        "You are an expert information engineering researcher focused on generating "
        "high-quality, specific, testable research directions and method candidates for "
        "{disease_name}. Prefer ideas grounded in strong prior work, clear evaluation plans, "
        "available datasets or benchmarks, and plausible technical contribution."
    )
    prompts.candidate_query_generation_content_message = (
        "Return a list of {double_queries} queries (separated by <>) that would be useful "
        "for background research toward this goal: {candidate_generation_goal}. The broader "
        "topic is {disease_name}. The queries should cover algorithms, architectures, datasets, "
        "evaluation metrics, baselines, ablations, deployment constraints, failure modes, and "
        "open research gaps. Generate {num_queries} broader survey-style queries and "
        "{num_queries} targeted method/evaluation queries."
    )
    prompts.candidate_generation_system_message = (
        "You are an expert information engineering researcher. Your task is to generate "
        "exactly {num_candidates} novel, testable research directions or method candidates "
        "for {disease_name}, based on the provided research goal and background literature.\n\n"
        "For each candidate, provide:\n"
        "1. `candidate`: A concise name for the research direction, method, system idea, "
        "benchmark, or evaluation approach.\n"
        "2. `hypothesis`: A specific, testable research hypothesis explaining why this idea "
        "could improve understanding or performance for {disease_name}.\n"
        "3. `reasoning`: Detailed reasoning grounded in prior work, expected contribution, "
        "datasets, metrics, baselines, feasibility, risks, and novelty.\n\n"
        "Output Format Specification (Strict Adherence Required):\n"
        "Generate exactly {num_candidates} blocks in this format:\n\n"
        "<CANDIDATE START>\n"
        "CANDIDATE: <candidate name>\n"
        "HYPOTHESIS: <testable hypothesis>\n"
        "REASONING: <detailed reasoning>\n"
        "<CANDIDATE END>"
    )
    prompts.candidate_generation_user_message = (
        "Generate exactly {num_candidates} distinct and rigorous proposals for research "
        "directions, method candidates, system designs, or benchmark ideas for {disease_name}. "
        "Here is relevant background information that can guide your proposals:\n"
        "{therapeutic_candidate_review_output}\n"
    )
    prompts.candidate_lit_review_direction_prompt = (
        "You are an information engineering research lead evaluating proposed research "
        "directions for {disease_name}. Given the following candidate idea, perform a "
        "comprehensive literature review assessing technical feasibility, novelty, datasets, "
        "metrics, baselines, reproducibility, and likely impact."
    )
    prompts.candidate_report_format = (
        "Provide your response in the following format, like an evaluation for a research proposal:\n"
        "Overview of Research Candidate: Explain the proposed method, system, benchmark, "
        "dataset, or research direction.\n"
        "Prior Work and Context: Summarize related work and how this candidate connects to "
        "or differs from existing approaches for {disease_name}.\n"
        "Technical Hypothesis: Explain the mechanism by which the candidate is expected to "
        "improve performance, understanding, robustness, usability, scalability, or evaluation quality.\n"
        "Evaluation Plan: Identify datasets, baselines, metrics, ablations, and reproducibility checks.\n"
        "Overall Evaluation: Discuss strengths, weaknesses, feasibility, novelty, and risks."
    )
    prompts.candidate_ranking_system_prompt = (
        "You are an experienced information engineering program committee member. Compare "
        "two research candidate proposals for {disease_name}. Select the one with the strongest "
        "combination of novelty, technical plausibility, evidence, evaluation rigor, feasibility, "
        "reproducibility, and potential research impact."
    )
    prompts.experimental_insights_appendage = (
        "Prior empirical studies have been conducted toward {candidate_generation_goal}. "
        "Summary: {experimental_insights_analysis_summary}. Relevant technical insights: "
        "{experimental_insights_mechanistic_insights}. Open questions: "
        "{experimental_insights_questions_raised}."
    )
    prompts.experimental_insights_for_candidate_generation = (
        "Prior empirical studies have been conducted toward {candidate_generation_goal}. "
        "Summary: {experimental_insights_analysis_summary}. Relevant technical insights: "
        "{experimental_insights_mechanistic_insights}. Open questions: "
        "{experimental_insights_questions_raised}."
    )
    return prompts


class AgentConfig(BaseModel):
    assay_lit_search_agent: JobNames = Field(
        default=JobNames.CROW,
        description="Agent to use for literature search during assay idea generation.",
    )
    assay_hypothesis_report_agent: JobNames = Field(
        default=JobNames.CROW,
        description="Agent to use for generating detailed reports on assay hypotheses.",
    )
    candidate_lit_search_agent: JobNames = Field(
        default=JobNames.CROW,
        description=(
            "Agent to use for literature search during therapeutic candidate idea"
            " generation."
        ),
    )
    candidate_hypothesis_report_agent: JobNames = Field(
        default=JobNames.FALCON,
        description=(
            "Agent to use for generating detailed reports on therapeutic candidates."
        ),
    )


class RobinConfiguration(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    prompts: Prompts = Field(default_factory=Prompts)
    num_queries: int = Field(
        default=3,
        description=(
            "Number of queries to generate for each step, more means more data but also"
            " more cost."
        ),
    )
    num_assays: int = Field(default=3, description="Number of assay to generate.")
    num_candidates: int = Field(
        default=5, description="Number of candidates to generate for each query."
    )
    application_domain: str = Field(
        default_factory=lambda: os.getenv("ROBIN_APPLICATION_DOMAIN", "biomedical"),
        description="Application domain: 'biomedical' or 'information_engineering'.",
    )
    research_topic: str | None = Field(
        default=None,
        description="Research topic to use instead of disease_name for non-biomedical domains.",
    )
    disease_name: str = Field(
        default="input_disease",
        description="Biomedical disease name or, for non-biomedical domains, research topic.",
    )
    run_folder_name: str | None = Field(
        default=None,
        description=(
            "Name of the folder where results will be stored. "
            "If not provided or None, it will be auto-generated "
            "using the disease_name and the timestamp."
        ),
    )
    edison_api_key: str = "insert_edison_api_key_here"
    llm_name: str = Field(default_factory=get_default_llm_name)
    llm_config: dict | None = Field(default_factory=get_default_llm_config)
    agent_settings: AgentConfig = Field(default_factory=AgentConfig)
    _edison_client: EdisonClient | None = PrivateAttr(default=None)
    _llm_client: LiteLLMModel | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def set_run_folder_name_default(self) -> "RobinConfiguration":
        if self.research_topic:
            self.disease_name = self.research_topic
        if self.application_domain == "information_engineering":
            self.prompts = build_information_engineering_prompts()
        if self.run_folder_name is None:
            disease_part = self.disease_name[:70].replace(" ", "_")
            timestamp_part = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.run_folder_name = f"{disease_part}_{timestamp_part}"
        return self

    @property
    def edison_client(self) -> EdisonClient:
        if self._edison_client is None:
            api_key = os.getenv("EDISON_API_KEY") or self.edison_api_key
            if not api_key:
                raise ValueError(
                    "Edison API key is not set. Please provide it in the"
                    " configuration or set EDISON_API_KEY env variable."
                )
            self._edison_client = EdisonClient(api_key=api_key)
        return self._edison_client

    @property
    def llm_client(self) -> LiteLLMModel:
        if self._llm_client is None:
            self._llm_client = LiteLLMModel(name=self.llm_name, config=self.llm_config)
        return self._llm_client

    def set_llm_model(self, model_name: str) -> None:
        self.llm_name = model_name
        if self.llm_config is None:
            self.llm_config = get_default_llm_config()
        self.llm_config["model_list"][0]["model_name"] = model_name
        self.llm_config["model_list"][0]["litellm_params"]["model"] = model_name
        self.llm_config["model_list"][0]["litellm_params"]["api_key"] = get_llm_api_key(
            model_name
        )
        self._llm_client = None

    def get_da_client(self):
        from .multitrajectory_runner import MultiTrajectoryRunner

        return MultiTrajectoryRunner(configuration=self)

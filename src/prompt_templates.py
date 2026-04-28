"""
Prompt Templates -- Week 4 (RAG)

CO-STAR framework and Week 4-specific prompt templates for grounded
question answering, query rewriting, and retrieval evaluation.
"""

from typing import Optional


class COSTARTemplate:
    """CO-STAR prompt framework template"""

    @staticmethod
    def build(
        context: str,
        objective: str,
        style: str = "professional",
        tone: str = "helpful",
        audience: str = "general",
        response_format: str = "text"
    ) -> str:
        """Build a CO-STAR prompt."""
        return f"""# Context
{context}

# Objective
{objective}

# Style
{style}

# Tone
{tone}

# Audience
{audience}

# Response Format
{response_format}
"""

    @staticmethod
    def build_system(
        style: str = "professional",
        tone: str = "helpful",
        response_format: str = "text"
    ) -> str:
        """Build a system prompt with S, T, R components."""
        return f"""You are a helpful AI assistant.

Style: {style}
Tone: {tone}
Response Format: {response_format}

Follow these guidelines in all responses."""


class PromptLibrary:
    """Library of pre-built prompts for Week 3 topics"""

    RESEARCH_ASSISTANT = COSTARTemplate.build(
        context="You are helping a researcher gather and analyze information.",
        objective="Find relevant information and synthesize it clearly.",
        style="academic but accessible",
        tone="objective and thorough",
        audience="researcher or student",
        response_format="structured with sources cited"
    )

    DATA_PIPELINE_EXPERT = COSTARTemplate.build(
        context="You are a data engineering expert helping design training data pipelines.",
        objective="Advise on data collection, cleaning, and quality for LLM pretraining.",
        style="practical and actionable",
        tone="direct and expert",
        audience="ML engineer or data scientist",
        response_format="structured recommendations with trade-offs"
    )

    OCR_ANALYST = COSTARTemplate.build(
        context="You are an OCR and document processing expert analyzing extraction results.",
        objective="Compare OCR outputs, identify errors, and suggest improvements.",
        style="technical and precise",
        tone="analytical and constructive",
        audience="data engineer working with document pipelines",
        response_format="comparison table followed by recommendations"
    )

    VOICE_AGENT_ARCHITECT = COSTARTemplate.build(
        context="You are designing a real-time voice AI agent with ASR, LLM, and TTS components.",
        objective="Help architect a low-latency voice pipeline with conversation memory.",
        style="systems architecture with clear component boundaries",
        tone="practical and performance-aware",
        audience="ML engineer building voice applications",
        response_format="architecture diagram (text), component specs, latency analysis"
    )

    DATA_QUALITY_REVIEWER = COSTARTemplate.build(
        context="You are reviewing a pretraining dataset for quality issues.",
        objective="Identify data quality problems and suggest filtering strategies.",
        style="thorough and systematic",
        tone="objective and detail-oriented",
        audience="data scientist building LLM training data",
        response_format="issue list with severity, examples, and recommended fixes"
    )

    TUTOR = COSTARTemplate.build(
        context="You are teaching a concept to a student.",
        objective="Explain clearly and verify understanding.",
        style="educational and patient",
        tone="encouraging and supportive",
        audience="student or learner",
        response_format="explanations with examples and follow-up questions"
    )

    @classmethod
    def get_template(cls, name: str) -> Optional[str]:
        """Get a template by name."""
        return getattr(cls, name.upper(), None)

    @classmethod
    def list_templates(cls) -> list:
        """List all available templates."""
        return [attr for attr in dir(cls)
                if not attr.startswith('_') and attr.isupper()]

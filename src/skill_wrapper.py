"""Week 5: gbrain-inspired skill wrapper for fine-tuned model routing."""

from __future__ import annotations

from typing import Callable


class FineTunedSkill:
    """gbrain-inspired skill wrapper that wraps a fine-tuned model as a named skill."""

    def __init__(
        self,
        name: str,
        description: str,
        model_fn: Callable[[str], str],
        match_keywords: list[str] | None = None,
    ):
        self.name = name
        self.description = description
        self.model_fn = model_fn
        self.match_keywords = [kw.lower() for kw in (match_keywords or [])]
        print(f"[skill_wrapper] Registered skill '{self.name}' with {len(self.match_keywords)} keywords")

    def matches(self, query: str) -> bool:
        """Return True if this skill should handle the query (keyword matching + heuristics)."""
        query_lower = query.lower()

        if any(kw in query_lower for kw in self.match_keywords):
            return True

        desc_words = [w.lower() for w in self.description.split() if len(w) > 4]
        if any(word in query_lower for word in desc_words):
            return True

        return False

    def invoke(
        self,
        query: str,
        retrieval_ctx: str = "",
        max_tokens: int = 300,
    ) -> str:
        """Run the fine-tuned model with optional RAG context. Return answer string."""
        print(f"[skill_wrapper] Skill '{self.name}' invoked for query: {query[:60]}...")

        if retrieval_ctx:
            prompt = (
                f"Use the following context to help answer the question.\n\n"
                f"Context:\n{retrieval_ctx}\n\n"
                f"Question: {query}\n\n"
                f"Answer:"
            )
        else:
            prompt = f"Question: {query}\n\nAnswer:"

        try:
            response = self.model_fn(prompt)
            print(f"[skill_wrapper] Skill '{self.name}' returned {len(response)} chars")
            return response
        except Exception as e:
            error_msg = f"[skill_wrapper] Skill '{self.name}' failed: {e}"
            print(error_msg)
            return error_msg

    def __repr__(self) -> str:
        keywords_preview = ", ".join(self.match_keywords[:5])
        if len(self.match_keywords) > 5:
            keywords_preview += f", ... (+{len(self.match_keywords) - 5} more)"
        return (
            f"FineTunedSkill(name='{self.name}', "
            f"keywords=[{keywords_preview}], "
            f"description='{self.description[:50]}...')"
        )


class SkillResolver:
    """Simple RESOLVER: route a query to the best matching skill."""

    def __init__(self, skills: list[FineTunedSkill]):
        self.skills = skills
        print(f"[skill_wrapper] SkillResolver initialized with {len(self.skills)} skill(s)")

    def resolve(self, query: str) -> FineTunedSkill:
        """Return the best matching skill for query."""
        print(f"[skill_wrapper] Resolving query: {query[:60]}...")

        # Count keyword hits per skill for ranking
        best_skill: FineTunedSkill | None = None
        best_score = -1

        for skill in self.skills:
            query_lower = query.lower()
            score = sum(1 for kw in skill.match_keywords if kw in query_lower)

            # Bonus for description word matches
            desc_words = [w.lower() for w in skill.description.split() if len(w) > 4]
            score += 0.5 * sum(1 for w in desc_words if w in query_lower)

            if score > best_score:
                best_score = score
                best_skill = skill

        if best_skill is None:
            if not self.skills:
                raise ValueError("[skill_wrapper] No skills registered in SkillResolver")
            best_skill = self.skills[0]
            print(f"[skill_wrapper] No match found, defaulting to first skill: '{best_skill.name}'")
        else:
            if best_score > 0:
                print(f"[skill_wrapper] Resolved to skill '{best_skill.name}' (score={best_score:.1f})")
            else:
                print(f"[skill_wrapper] No keyword match, falling back to '{best_skill.name}'")

        return best_skill

    def dispatch(self, query: str, retrieval_ctx: str = "") -> str:
        """Resolve + invoke in one step."""
        skill = self.resolve(query)
        return skill.invoke(query, retrieval_ctx=retrieval_ctx)

    def show_skills(self) -> None:
        """Print all registered skills and their match keywords."""
        print(f"[skill_wrapper] Registered skills ({len(self.skills)} total):")
        for i, skill in enumerate(self.skills):
            keywords_str = ", ".join(skill.match_keywords) if skill.match_keywords else "(none)"
            print(f"  [{i + 1}] {skill.name}")
            print(f"       Description: {skill.description}")
            print(f"       Keywords: {keywords_str}")


def make_resume_skill(model_fn: Callable[[str], str]) -> FineTunedSkill:
    """Create a pre-configured FineTunedSkill for resume Q&A."""
    return FineTunedSkill(
        name="resume_qa",
        description="Answers questions about professional experience, skills, education, and career history from a resume.",
        model_fn=model_fn,
        match_keywords=[
            "resume",
            "experience",
            "skills",
            "education",
            "career",
            "job",
            "work",
            "background",
            "qualification",
            "degree",
            "university",
            "company",
            "position",
            "role",
            "project",
            "achievement",
            "certification",
            "profile",
            "candidate",
        ],
    )

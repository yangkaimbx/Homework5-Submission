# RESOLVER.md — Skill Routing Guide

## hw5-resume-skill
**When to use:** Queries about resume content, work history, skills, education, career goals  
**Keywords:** resume, experience, job, background, skills, education, worked, built, achieved  
**Example queries:**
- "What languages does Scott know?"
- "Tell me about Scott's experience at [company]"
- "What is Scott's highest degree?"
- "What projects has Scott built?"
- "What kind of roles is Scott targeting?"

**Model:** hw5-finetuned (Qwen2.5-0.5B-Instruct, fine-tuned on synthetic resume Q&A)  
**RAG:** Yes — retrieves top-3 chunks from sample_resume.pdf before generating  

## Fallback
If no skill matches, route to the base LLM (qwen3.5:27b via Ollama).  
Trigger condition: query does not contain resume-related keywords AND no skill confidence > 0.5.

# Week 5 Project Update — Kai Yang

## What I built this week
This week I built a fine-tuned resume assistant. I trained a LoRA adapter on Qwen2.5-0.5B-Instruct using synthetic resume Q&A data, tried a small DPO preference-tuning step, merged the model, converted it to
GGUF, and served it locally with Ollama as hw5-finetuned. I also wrapped it as a gbrain-style skill and added a general Q&A skill so the resolver can route resume questions and normal knowledge questions
separately.

## How Week 5 connects to Week 4 (RAG + fine-tuning)
Week 4 handled retrieval: the RAG pipeline pulls relevant chunks from the resume PDF. Week 5 adds the fine-tuned model on top, so the assistant can use that retrieved context and answer in a more polished,
resume-focused style. RAG helps keep the facts grounded, while fine-tuning helps with tone and consistency.

## What surprised me most about fine-tuning
I was surprised that a lower training loss did not automatically mean the answers were fully reliable. The fine-tuned model sounded more like a resume assistant, but it could still hallucinate when the context
was weak. My eval improved only a little, from 1.80 for the baseline to 2.00 for the fine-tuned model, which showed me that good data and retrieval matter a lot.

## What I would improve with more compute/time
I would use more verified training examples, add a real validation set, and make the DPO dataset much larger. I would also make RAG required for resume questions and improve the resolver so it can be more
confident about when to use the resume skill versus the general Q&A skill.
# Week 5: Fine-Tuning LLMs -- Homework Reflection

**Student Name:** Kai Yang

**Path Selected:** B (HF+TRL)

---


## Notebook 01: Environment Setup & Path Selection

**Completed:** 2026-05-03 23:51:33

### TODO 1

The baseline response was much longer. After the system prompt was changed, the same user prompt produced one compact sentence. This shows that a chat-tuned model treats the system prompt as a high-priority instruction that shapes tone, length, and formatting even when the user prompt stays the same.


### TODO 2

A base model is trained mainly on large-scale raw text datasets such as web pages, books, articles etc., where the objective is next-token prediction. An instruct model starts from a base model and is further trained on instruction-response data, such as question-answer pairs, chat conversations, etc. 
I would use a base model when I want maximum flexibility for continued pretraining or custom fine-tuning, but I would use an instruct model for chat, question answering, and task-following.
Because Qwen2.5-0.5B-Instruct has learned to answer instructions while Qwen2.5-0.5B may only continue the prompt.

---

## Notebook 02: Data Formats & Chat Templates

**Completed:** 2026-05-04 00:31:34

### TODO 1

My name has the same token counts (3) as its lowercase version, maybe because it is simple. "Python"'s count also remains the same. 
(a)For my name, changing capitalization did not change the count, but spacing can change the count because spaces may be merged into neighboring tokens or become separate tokens. 
(b)"Python" and "python" are not the same token because they use different token IDs and decoded pieces. 
(c)BPE learns merges from exact byte sequences, so uppercase and lowercase are different inputs and can learn different frequency-based merges.


### TODO 2

I chose questions about my background, a technical project, and how I learn new tools because these are realistic recruiter questions that evaluate fit, experience, ownership, and adaptability. A recruiter would ask the background question to understand my career direction, the project question to assess technical impact, and the learning question to see how I handle unfamiliar technologies. 
The system prompt tells the assistant to answer as a professional job candidate in first person, which makes the responses concise, resume-focused, and appropriate for a recruiter conversation.

---

## Notebook 03: Synthetic Data Generation

**Completed:** 2026-05-04 01:20:08

### TODO 1

The three manual seed questions I added were: 
"What is your preferred work environment - remote, hybrid, or in-office?", 
"Can you describe a time when you had to learn a new technology quickly?", and 
"What salary range are you targeting?" 
The auto-generated seeds mainly captured resume-grounded facts like roles, skills, education, and accomplishments, but they missed interview-style topics such as work preferences, learning behavior, and compensation expectations. These gaps come mostly from the corpus and prompt template: a resume usually does not include salary targets or preferred work setup, and a corpus-grounded LLM tends to avoid inventing personal preferences unless explicitly asked to generate those edge-case topics.

### TODO 2

EvolInstruct is a technique from WizardLM where an LLM iteratively rewrites simple instructions into more complex ones using operations like adding constraints, increasing reasoning depth, or making the task more specific. In this notebook, we generated data grounded in a resume using a seed -> expand -> filter pipeline, so the goal was factual coverage of a corpus rather than open-ended instruction difficulty evolution.

---

## Notebook 04: LoRA, QLoRA, DoRA & Beyond

**Completed:** 2026-05-04 07:05:48

### TODO 1: Absolute Parameter Count

# Q1: How many absolute params?
# absolute,trainable = 494,032,896 * 0.008 = 3,952,263

# Q2: Compare to full fine-tune
# full,params = total,params = 494,032,896
# ratio = 494,032,896 / 3,952,263 = 125

# Q3: Memory impact — Adam optimizer stores (param + grad + m + v) = 4 × params × 4 bytes (fp32)
# lora,optimizer,bytes = 4 * 3,952,263 * 4 =  63,236,208 = 63 MB
# full,optimizer,bytes = 4 * 494,032,896 * 4 =  7,904,526,336  =7.9 GB

This matters on a 16GB Mac because Adam optimizer memory scales with the number of trainable parameters. The optimizer-related memory is about 60 MiB for LoRA versus about 7.36 GiB for full fine-tuning. Saving that much memory makes training much more feasible on limited RAM/VRAM.


### TODO 2: Choosing a Production Variant

I would choose RSLoRA because Qwen2.5-0.5B is small enough that a lightweight, stable LoRA variant is more practical than adding extra overhead or complexity. 
1. RSLoRA is the better fit than DoRA under these constraints because it has almost no overhead, while DoRA's extra per-step cost is worth it mainly when maximum instruction-following quality matters more than training speed. 
2. PiSSA's faster convergence could help within a 2-hour budget, but for this small resume Q&A assistant I would prioritize RSLoRA.
3. I would pair it with r=16 and alpha=32 because it gives a strong quality/memory tradeoff on a 16GB Mac.

---

## Notebook 05: Supervised Fine-Tuning with TRL

**Completed:** 2026-05-04 08:44:50

### TODO 1: Extend the Smoke Test (max_steps=50)

After extending max_steps to 50, the logged training loss continued decreasing from 1.554507 at step 20 to 0.108949 at step 50. The decrease was steady, though the improvements became smaller near the end, suggesting the run was starting to approach a plateau. This means 20 steps was useful only as a pipeline smoke test, not as a sufficient training run; in real runs, max_steps should be chosen by watching the loss curve rather than assuming an early step count is enough.


### TODO 2: Overfitting Risk

Overfitting in SFT happens when the model learns to reproduce the small training set instead of generalizing to new instructions. A 50-pair dataset trained for 10 epochs is risky because the model sees the same limited examples repeatedly, which encourages memorization. In practice, I would expect brittle responses to rephrased prompts, verbatim regurgitation of training answers, and possible hallucination of training-specific facts.

---

## Notebook 06: Preference Tuning: DPO, KTO & GRPO

**Completed:** 2026-05-04 11:58:45

### TODO 1: Manual Preference Pairs

I authored three manual DPO preference pairs covering Scott's LLM application stack, his language-model fine-tuning experience, and his career goals for the next year.
For the LLM application stack pair, I used a "too vague" rejected answer. The rejected response says he uses various AI tools but does not mention concrete technologies like Claude API, FAISS, TRL, MLX-LM, or Ollama, which makes it much less useful for resume Q&A.
For the fine-tuning pair, I used a "contradicts facts" rejected answer. It falsely claims Scott fine-tuned GPT-4 from scratch with full RLHF at OpenAI, which would be harmful in production because a resume assistant needs to be accurate and should not inflate credentials.
For the career goals pair, I used a "wrong tone" rejected answer. The rejected response sounds overly casual and compensation-focused, while a production resume assistant should give professional, recruiter-appropriate answers that emphasize growth, ML engineering, and production LLM systems.


### TODO 2: KTO vs DPO Scenario

I would choose KTO over DPO for an e-commerce customer support assistant. In that setting, customers ask one question about orders, refunds or returns etc, receive one answer, and then mark whether the response resolved their issue.

This produces binary labels instead of chosen/rejected pairs because the customer is judging a real interaction, not comparing two alternative model completions. A resolved or agent-approved answer becomes label=True, while a not-resolved or agent-rejected answer becomes label=False.

To collect the data, I would log the customer question and assistant response, capture the resolved/not-resolved feedback or agent review result, and export each row as {"prompt", "completion", "label"}. KTO fits better here because it can train directly on low-friction production feedback without requiring human annotators to create paired responses.

---

"""
Part C -- Step 1: Prompt Templates
CS4241 Introduction to Artificial Intelligence 2026
Student: [Your Name] | Index: [Your Index Number]

Defines FOUR prompt template iterations for the RAG pipeline.
Each template targets a specific generation quality property:

  T1 -- Baseline         : minimal structure, no constraints
  T2 -- Structured       : role + context injection + answer prefix
  T3 -- Hallucination Guard : explicit "only use context" + refusal rule
  T4 -- Chain-of-Thought : step-by-step reasoning before final answer

Design Rationale
----------------
Prompt engineering is iterative: start minimal (T1), add structure (T2),
add safety constraints (T3), then add reasoning scaffolding (T4).
Each iteration is tested on the same queries in Step 3 so that output
differences can be measured and attributed to specific template changes.

Hallucination Control Strategy (T3, T4)
----------------------------------------
1. Role constraint: "You are a factual assistant using ONLY the context."
2. Explicit refusal instruction: "If not in context, say 'I don't know'."
3. Evidence anchoring: "Cite the source (election/budget) when possible."
These three mechanisms together reduce the model's tendency to blend
retrieved knowledge with its parametric (pre-trained) knowledge.

Context placeholder : {context}   -- filled by context window manager
Query placeholder   : {query}     -- filled at query time
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptTemplate:
    """A named, versioned prompt template with metadata."""
    name:        str
    version:     str
    description: str
    system_msg:  Optional[str]   # None -> no system message (some models ignore it)
    user_template: str           # must contain {context} and {query}

    def build(self, context: str, query: str) -> dict:
        """
        Fill the template and return a messages list suitable for
        the Groq chat-completions API.

        Returns
        -------
        {"system": str | None, "user": str}
        """
        user_text = self.user_template.format(
            context=context.strip(),
            query=query.strip(),
        )
        return {
            "system": self.system_msg,
            "user":   user_text,
        }

    def to_messages(self, context: str, query: str) -> list[dict]:
        """Return OpenAI-style messages list (for the Groq client)."""
        filled = self.build(context, query)
        msgs = []
        if filled["system"]:
            msgs.append({"role": "system", "content": filled["system"]})
        msgs.append({"role": "user", "content": filled["user"]})
        return msgs


# ==============================================================================
# TEMPLATE 1 -- Baseline (minimal)
# ==============================================================================
T1_BASELINE = PromptTemplate(
    name        = "T1_Baseline",
    version     = "1.0",
    description = (
        "Minimal prompt: raw context block followed by the question. "
        "No role, no constraints, no hallucination guard. "
        "Establishes a lower-bound quality baseline."
    ),
    system_msg  = None,
    user_template = """\
Context:
{context}

Question: {query}

Answer:""",
)


# ==============================================================================
# TEMPLATE 2 -- Structured (role + format)
# ==============================================================================
T2_STRUCTURED = PromptTemplate(
    name        = "T2_Structured",
    version     = "1.0",
    description = (
        "Adds a system role ('expert assistant for Academic City') and "
        "a structured user prompt with labelled sections. "
        "Improves formatting and relevance over T1 without adding "
        "strict hallucination controls."
    ),
    system_msg  = (
        "You are an expert AI assistant for Academic City, Ghana. "
        "Your role is to answer questions accurately and concisely "
        "based on provided reference documents."
    ),
    user_template = """\
## Retrieved Reference Documents
{context}

## Question
{query}

## Answer
Provide a clear, concise answer based on the reference documents above:""",
)


# ==============================================================================
# TEMPLATE 3 -- Hallucination Guard (strict grounding)
# ==============================================================================
T3_HALLUCINATION_GUARD = PromptTemplate(
    name        = "T3_Hallucination_Guard",
    version     = "1.0",
    description = (
        "Strict grounding instructions: only use provided context, "
        "refuse to speculate, cite the source dataset explicitly. "
        "Targets minimising hallucination rate at potential cost of "
        "answer completeness for under-retrieved queries."
    ),
    system_msg  = (
        "You are a factual, grounded AI assistant for Academic City. "
        "RULES -- follow these strictly:\n"
        "1. Answer ONLY using information from the CONTEXT section below.\n"
        "2. Do NOT use any external or pre-trained knowledge.\n"
        "3. If the answer is not present in the context, respond exactly:\n"
        "   'INSUFFICIENT CONTEXT: I cannot answer this from the provided documents.'\n"
        "4. When possible, indicate which source (Election data or Budget document) "
        "your answer comes from.\n"
        "5. Do not speculate, estimate, or make assumptions."
    ),
    user_template = """\
CONTEXT (retrieved documents):
---
{context}
---

USER QUESTION: {query}

GROUNDED ANSWER (context only, cite source where possible):""",
)


# ==============================================================================
# TEMPLATE 4 -- Chain-of-Thought (reasoning before answer)
# ==============================================================================
T4_CHAIN_OF_THOUGHT = PromptTemplate(
    name        = "T4_Chain_of_Thought",
    version     = "1.0",
    description = (
        "Instructs the model to reason step-by-step before producing "
        "a final answer. Chain-of-thought (CoT) prompting improves "
        "complex multi-step questions (e.g., budget calculations, "
        "multi-year election trends) by externalising the reasoning "
        "process, making errors detectable and correctable."
    ),
    system_msg  = (
        "You are a careful, analytical AI assistant for Academic City. "
        "When answering questions:\n"
        "1. First, identify relevant information from the context.\n"
        "2. Reason step-by-step using only the provided context.\n"
        "3. State any limitations or gaps in the available information.\n"
        "4. Provide a concise, well-cited final answer.\n"
        "Do not use knowledge outside the provided context."
    ),
    user_template = """\
## Context Documents
{context}

## Question
{query}

## Reasoning Process
Step 1 - Identify relevant context passages:
Step 2 - Extract key facts:
Step 3 - Synthesise the answer:

## Final Answer""",
)


# ==============================================================================
# Registry -- import this in other modules
# ==============================================================================

ALL_TEMPLATES: list[PromptTemplate] = [
    T1_BASELINE,
    T2_STRUCTURED,
    T3_HALLUCINATION_GUARD,
    T4_CHAIN_OF_THOUGHT,
]

TEMPLATE_MAP: dict[str, PromptTemplate] = {t.name: t for t in ALL_TEMPLATES}


# ==============================================================================
# Demo (run as script)
# ==============================================================================

if __name__ == "__main__":
    SAMPLE_CONTEXT = (
        "Year: 2020. Candidate: Nana Addo Dankwa Akufo-Addo. Party: NPP. "
        "Votes: 6,730,413. Votes(%): 51.59%.\n"
        "Year: 2020. Candidate: John Dramani Mahama. Party: NDC. "
        "Votes: 6,214,889. Votes(%): 47.66%."
    )
    SAMPLE_QUERY = "Who won the 2020 Ghana presidential election and by how many votes?"

    print("=" * 65)
    print("  PART C -- STEP 1: PROMPT TEMPLATES PREVIEW")
    print("=" * 65)

    for tmpl in ALL_TEMPLATES:
        print(f"\n  Template : {tmpl.name}")
        print(f"  Desc     : {tmpl.description[:80]}...")
        msgs = tmpl.to_messages(SAMPLE_CONTEXT, SAMPLE_QUERY)
        for m in msgs:
            role    = m["role"].upper()
            preview = m["content"][:200].replace("\n", " ")
            print(f"  [{role}] {preview}...")
        print()

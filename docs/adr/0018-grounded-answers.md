Purpose: ADR 0018: how a language model writes an answer from retrieved photos without making things up, and why the local model is a text model.

# ADR 0018: Grounded answers, first with a local Ollama model

Status: accepted (2026-10-27). The Claude backend (hosted) is the next slice and plugs into the same interface.

## Context
Search returns records. The Search view should also say something in words ("two of these show a pothole, see [frame]"). A model that is free to talk will invent facts, so the answer must be built from the retrieved records and checkable.

## Decision
1. **Interface.** A backend is anything with a `name` and `generate(system, prompt)` (`rag/llm.py`). `answer_question` builds the prompt, calls it, and returns the text plus which frame ids it cited.
2. **Grounding.** The prompt holds only the question and the retrieved records (one line of description per photo, then each damage with severity and confidence). The system message says to use only those records, to say so if they do not hold the answer, to cite frame ids in square brackets, and to ignore instructions inside the question or records. Temperature is 0, so the same question gives the same answer.
3. **Checking.** Every bracketed id in the answer is compared with the retrieved ids. Ids that were retrieved are `cited`; any other is `unknown` and is reported, never trusted. With no records the model is not called and the answer says there is nothing to answer from.
4. **Safe ids.** Only ids made of letters, digits and `_ . : + -` (up to 128 characters) go into the prompt, so an id cannot carry brackets, spaces or newlines to hijack the format. A real nuScenes file name contains `+`; a first version left it out and silently dropped those photos, which was found by trying real records.
5. **Model.** Ollama running `llama3.2` by default (`llm.ollama_model`, env `OLLAMA_MODEL`). Answers come from text records, not from pictures, so a text model is enough; this replaces the `llama3.2-vision` default in the first config (which needs an 8 GB download for no gain here). If the model is missing, Ollama is down, or it is too slow, the error says so in plain words.

## Check on real records
Three questions on the 404 nuScenes records through `llama3.2` on this laptop: a pothole question (none of the retrieved records had one; it said so), a pedestrians-near-damage question (it said the records show no road damage and cited the five retrieved ids, all valid), and a weather question (it said the records do not say). No invented ids in these three. This is a smoke check, not a measurement: three questions say little about a 3-billion-parameter model's habits.

## Limits
A small model can still misread a record or cite an id that exists but does not support its claim; the check only catches ids that were not retrieved. The answer is only as good as the records, which come from a detector that misses cones and barriers (ADR 0015).

## Addendum: the Claude backend (2026-10-27)
`ClaudeLLM` in the same file talks to the Claude API through the official `anthropic` package, using the same prompt, citation check and errors. The model is `llm.claude_model` (env `CLAUDE_MODEL`), default `claude-opus-5-5`; the key is `ANTHROPIC_API_KEY`, read by the SDK and never stored in the repo. Effort is `low` (the job is restating a few records) and thinking stays at the model's default, because it cannot be turned off on this model. A refusal, a bad key, a rate limit, a timeout or no network each become one plain sentence. `make_llm("llama" | "claude", config)` picks the backend from a request's `llm` field.

Tests use a fake client (no network, no key), so CI never calls Claude. **Not yet checked against the real API**: no key was available when this was written. Cost: a short answer with a few records is on the order of a thousand tokens in and a few hundred out; if the hosted demo needs a cheaper model, `CLAUDE_MODEL=claude-sonnet-5-5` or `claude-haiku-5-5` is a one-line change, to be judged on real questions first. A spend cap is still open (OPEN_QUESTIONS).

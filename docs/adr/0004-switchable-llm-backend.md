Purpose: ADR 0004: Switchable LLM backend.

# ADR 0004: Switchable LLM backend

Status: accepted (owner approved)

## Context
Answers and reports need a vision-language model. LLaMA 3.2 Vision via Ollama is local, free and already used in defect-detector. The Claude API is hosted, gives better answers, and costs money; usage is billed separately from the Claude subscription. The hosted demo has a budget of about $20/month and no GPU.

## Decision
Support both behind an `LLMBackend` protocol (`OllamaBackend`, `ClaudeBackend`), selected by `LLM_BACKEND` (`ollama` or `claude`) in config or env. The request may also choose `llm: llama|claude`. Claude key from env only; monthly spend cap (`ANTHROPIC_MONTHLY_SPEND_CAP_USD`) with a hard cutoff and a fallback message. In CI the Claude backend is behind a mocked client. <!-- F:R-02 --> <!-- F:A-14 -->

## Alternatives considered
- Claude-only: cost, no offline mode. Rejected.
- Local-only: weaker hosted demo quality. Rejected.

## Consequences
Two code paths to test. Prompts must ground strictly in retrieved records and cite frame IDs; instructions inside records are not executed. Nightly live check runs with a cost cap.

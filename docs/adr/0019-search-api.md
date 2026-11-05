Purpose: ADR 0019: how /index, /query and /records work per visitor, how the models are chosen, and the limits that protect a public endpoint.

# ADR 0019: The search API

Status: accepted (2026-10-27). Contract v0.5.0.

## Context
The search core (ADR 0017) and grounded answers (ADR 0018) existed but the routes were stubs sharing one dictionary. A hosted demo needs per-visitor photos, cost control and abuse limits (docs/SECURITY_RAG.md).

## Decision
1. **Sessions.** The server issues an unguessable id in an HttpOnly `rh_session` cookie (`api/session.py`). The id is never read from a body or header; a cookie that is not a well-formed server id is replaced. Photos go to the session through `Index.add_upload`, which refuses the shared corpus; searches cover the corpus plus the caller's session. `/records/{id}` returns the caller's or the corpus record, and another session's record is the same 404 as a missing one. `/analyze` keeps its last records per session (bounded) so `/records` still works for them.
2. **Query.** Text is cleaned (`clean_question`) and embedded; it searches both vectors. A photo searches the photo vector. Both together search the photo vector with the photo and the description vector with the text. Filters are typed. A photo-only query returns the similar photos and calls no model.
3. **Models.** `llm: "claude"` tries `claude-haiku-5-5` first and `claude-sonnet-5-5` only when the first answer fails a check the code can make: the model failed or declined, was cut off, or cited an id that was not among the retrieved records. The model's own "I am unsure" is not used. `llm: "llama"` is one local Ollama model. If every model fails, or the spend limit is reached, the response still carries the photos plus a fixed `notice`.
4. **Spend.** Token counts from each Claude reply are priced (`rag/guards.py`) and added up; new Claude calls stop at the daily and monthly limits (`rag.daily_usd`, `rag.monthly_usd`, env `ANTHROPIC_DAILY_SPEND_CAP_USD`, `ANTHROPIC_MONTHLY_SPEND_CAP_USD`). This is an in-process estimate that resets on restart; the Anthropic Console limit is the real cap.
5. **Limits.** Per session and per client address (`request.client`; the code never reads `X-Forwarded-For` itself, but uvicorn fills `request.client` from it when the sender is the local website server, its default, so a header that reaches the API can matter; see the check on 2026-11-04 under Limits of this design): 20 queries and 6 uploads a minute, 429 with `Retry-After`. At most 3 model calls at once, then a 503. At most 100 photos per session; a session idle for two hours is deleted; at most 500 sessions, least recently used first.
6. **Errors.** A model failure never reaches the visitor as text: the response says the answer service is unavailable, the reason goes to the log.
7. **Stubs.** With `USE_STUBS=1` a hash-based embedder replaces CLIP so the whole flow runs in CI; it knows nothing about meaning and is never used for a measurement.

## Limits of this design
- Rate limits, spend and sessions live in one process. Two server processes would each allow the full rate and keep their own spend and sessions; the hosted demo must run one.
- The address limit treats everyone behind one proxy as one visitor; on a host that proxies, the proxy's address is what is limited.
- Forged forwarding header, checked 2026-11-04 (release review finding F4). A local `next start` plus uvicorn let rotating `X-Forwarded-For` values dodge the address limit. On the private Space, 25 cookie-less requests with 25 different made-up values got 20 times 404 and then 429, so the hosted setup ignores the forged value. Not established: whether the limited address is each visitor's own or the Hugging Face proxy's (a shared bucket for everyone). A local run outside the Space is still dodgeable; the middleware that would fix it is not built because the hosted result does not need it.
- The Haiku-then-Sonnet rule has not been tuned on a question set; it has only been run on four questions. The escalation checks catch invented ids, not wrong claims that cite real ids.

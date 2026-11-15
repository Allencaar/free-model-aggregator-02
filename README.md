# Swift Relay 2 — a live free-model aggregator

One server that makes OpenRouter's fourteen free models a single endpoint, built for
make.fun's **free-model-aggregator-02** bounty.

A client points at this server and every completion it answers comes from a real free model
dialled through the grader's gateway:

- `POST /v1/chat/completions` — OpenAI-compatible, JSON and streaming
- `POST /v1/messages` — Anthropic-compatible, JSON and streaming (translated to Chat
  Completions upstream and back)
- `GET /v1/models` — the fourteen free models, in the shape of the caller's auth header

## How it routes

The trial brief lists the free models with the house's quality ratings and the six prompts'
requirements. For each prompt the server ranks every free model that meets the requirements
by rating and dials the best first; a refusal fails over to the next candidate, a model that
refuses twice in a row cools down, and the refusals a prompt recovered from are reported in
its answer's `x-makefun-upstream-failures` header. Every answer names the model that really
served it. All choices are a function of the brief, the requests and the gateway's answers —
no clock, no randomness — so a replay of a trial reproduces it byte for byte.

## Run

```
node main.ts          # serve a trial (MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE, OPENAI_BASE_URL set)
node main.ts --list   # print the catalog document the server registers
```

Node 22, standard library only. MIT licensed.

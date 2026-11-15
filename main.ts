// free-model-aggregator-02: one server, two client wires, and the best free model behind
// every answer.
//
// WHAT A TRIAL GIVES THIS SERVER
//   A brief at $MAKEFUN_PROMPT_FILE (the fourteen free OpenRouter models with the house's
//   ratings, and the six prompts' manifest in sending order) and an env at $MAKEFUN_ENV_URL.
//   The server builds one catalog from the brief, listens on a free port, POSTs /register with
//   that catalog, then answers the client's six prompts one at a time, dialling real free models
//   through the gateway at $OPENAI_BASE_URL (Chat Completions is the only route these ids have),
//   and polls GET /work every 100 ms, exiting 0 when it says done. `node main.ts --list` prints
//   the same catalog JSON and exits 0 before any server starts.
//
// HOW IT ROUTES (the score: quality 60, served 15, wire 10, failover 10, catalog 5)
//   - quality: every free model meeting the prompt's requirements is ranked by the brief's
//     rating_milli, best first, and the best is dialled first. Requirements are the manifest
//     row for the arriving prompt united with what the body itself asks for: tools when the
//     body declares tools, vision when it carries an image, JSON mode when it carries an
//     OpenAI response_format.
//   - failover: a refusal (any non-2xx, or a failed fetch) is recorded and the prompt moves to
//     the next candidate, best first. A model that has refused twice in a row is treated as
//     cooling down and skipped while any fresh candidate remains, so one rate-limited model
//     cannot tax every later prompt with a rediscovery call. At most MAX_ATTEMPTS calls are
//     spent on one prompt; the gateway caps a trial at 60 and the tie-break counts the calls
//     the provider answered, so calls are never speculated in parallel.
//   - the trail: refusals a prompt recovered from are reported on its answer in the
//     x-makefun-upstream-failures header, the only way the failover column can see them.
//   - wires: an OpenAI request rides upstream as it came, with only the model swapped (plus
//     stream_options.include_usage on a stream, which otherwise reports no usage); an Anthropic
//     request is translated through lib/wire.ts's canonical form. Every upstream answer is
//     buffered whole, read into the canonical answer, and written out in the client's wire with
//     the dialled model's id on it — the provider's own spelling drops the gateway's prefix.
//   - determinism: every choice is a function of the brief, the requests and the gateway's
//     answers seen so far. No wall clock and no randomness reach a request or an answer; the
//     only clock read is the per-call timeout, which decides nothing but when to give up
//     waiting.
//
// FILES
//   lib/models.ts   the brief's documents, the catalog, the requirement test.
//   lib/wire.ts     the canonical form, the translations, the answer writers.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/models.ts';
import {
  answerFromOpenAI,
  answerToAnthropic,
  answerToOpenAI,
  anthropicError,
  anthropicModelList,
  canonFromAnthropic,
  canonFromOpenAI,
  canonToOpenAI,
  hasImage,
  openAIError,
  openAIModelList,
  type CanonAnswer,
  type CanonRequest,
  type ClientWire,
  type ListedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one gateway call may take before the prompt fails over to the next candidate. */
const CALL_TIMEOUT_MS = 20000;
/** The most gateway calls one prompt may cost, first pick and failovers together. */
const MAX_ATTEMPTS = 4;
/** Consecutive refusals after which a model is skipped while a fresh candidate remains. */
const COOLDOWN_AFTER = 2;
/** The output cap named when a client body names none (the gateway refuses a call without one). */
const DEFAULT_MAX_TOKENS = 1024;
/** The gateway's own output cap: a client cap above it is refused, so it is clamped to it. */
const GATEWAY_MAX_TOKENS = 8192;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The response header that carries the refusals a served prompt recovered from. */
const FAILURES_HEADER = 'x-makefun-upstream-failures';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which manifest row the request now arriving is: the client sends the prompts in order. */
let manifestIndex = 0;
/** Serialize request handling so gateway calls never overlap. */
let queue: Promise<void> = Promise.resolve();

/** Per-model call state, learned from the gateway's answers. */
interface ModelState {
  calls: number;
  consecutiveRefusals: number;
}
const modelState = new Map<string, ModelState>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function stateOf(modelId: string): ModelState {
  let state = modelState.get(modelId);
  if (state === undefined) {
    state = { calls: 0, consecutiveRefusals: 0 };
    modelState.set(modelId, state);
  }
  return state;
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1`
 * on the trial network); a base that does not already end in `/v1` gets one.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models, as GET /v1/models names them. */
function servedModels(): ListedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * Every legal pick for one prompt, best first: free models meeting the requirements, ranked by
 * the brief's rating. Ties keep the brief's order. A model in cooldown (COOLDOWN_AFTER refusals
 * in a row) sorts behind every fresh model but stays in the list, so it is still tried when
 * nothing fresh remains.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  const fresh: ModelRow[] = [];
  const cooling: ModelRow[] = [];
  for (const model of freeModels(brief)) {
    if (!meetsRequirements(requires, capabilitiesOf(model))) continue;
    if (stateOf(model.id).consecutiveRefusals >= COOLDOWN_AFTER) cooling.push(model);
    else fresh.push(model);
  }
  const byRating = (a: ModelRow, b: ModelRow): number => b.rating_milli - a.rating_milli;
  fresh.sort(byRating);
  cooling.sort(byRating);
  return [...fresh, ...cooling];
}

/** The requirements of the request now arriving: the manifest's row united with the body's own. */
function requirementsFor(canon: CanonRequest): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  const requires: Requirement = { ...(entry?.requires ?? {}) };
  if (canon.tools.length > 0) requires.tools = true;
  if (hasImage(canon)) requires.vision = true;
  if (canon.jsonMode) requires.json = true;
  return requires;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendBody(res: ServerResponse, status: number, contentType: string, body: string, failures: FailureRow[]): void {
  const headers: Record<string, string | number> = { 'content-type': contentType, 'content-length': Buffer.byteLength(body) };
  if (contentType === 'text/event-stream') headers['cache-control'] = 'no-cache';
  if (failures.length > 0) headers[FAILURES_HEADER] = JSON.stringify(failures.slice(0, 8));
  res.writeHead(status, headers);
  res.end(body);
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  sendBody(res, status, 'application/json', JSON.stringify(body), []);
}

function sendError(res: ServerResponse, wire: ClientWire, status: number, type: string, code: string | null, message: string, failures: FailureRow[]): void {
  const body = wire === 'anthropic' ? anthropicError(type, message) : openAIError(message, type, code);
  sendBody(res, status, 'application/json', JSON.stringify(body), failures);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One refusal the gateway handed back, as the failover trail reports it. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** The refusal code a gateway error body names: its `error.code`, else its `error.type`, else ''. */
function errorCodeOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code.slice(0, 64);
      if (typeof value.error.type === 'string') return value.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: no code to name.
  }
  return '';
}

/** The upstream body for one candidate: the client's own body re-pointed, or the translation. */
function upstreamBody(clientWire: ClientWire, clientBody: Json, canon: CanonRequest, model: string): Json {
  if (clientWire === 'anthropic') return canonToOpenAI(canon, model);
  const body: Json = { ...clientBody, model };
  const cap = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  body.max_tokens = Math.min(cap ?? DEFAULT_MAX_TOKENS, GATEWAY_MAX_TOKENS);
  delete body.max_completion_tokens;
  if (canon.stream) {
    const options = isRecord(clientBody.stream_options) ? clientBody.stream_options : {};
    body.stream_options = { ...options, include_usage: true };
  }
  return body;
}

type Attempt =
  | { ok: true; answer: CanonAnswer }
  | { ok: false; failure: FailureRow; detail: string };

/** One candidate tried for one prompt: the gateway called once, its answer buffered whole. */
async function tryCandidate(candidate: ModelRow, clientWire: ClientWire, clientBody: Json, canon: CanonRequest): Promise<Attempt> {
  const body = upstreamBody(clientWire, clientBody, canon, candidate.id);
  let upstream: Response;
  try {
    upstream = await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${GATEWAY_KEY}`,
        ...(canon.stream ? { accept: 'text/event-stream' } : {}),
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    const state = stateOf(candidate.id);
    state.consecutiveRefusals += 1;
    return {
      ok: false,
      failure: { status: 503, code: 'upstream_unreachable', model: candidate.id },
      detail: (err as Error).message,
    };
  }
  stateOf(candidate.id).calls += 1;
  const text = await upstream.text();
  if (!upstream.ok) {
    const state = stateOf(candidate.id);
    state.consecutiveRefusals += 1;
    return {
      ok: false,
      failure: { status: upstream.status, code: errorCodeOf(text), model: candidate.id },
      detail: `HTTP ${upstream.status}: ${text.slice(0, 300)}`,
    };
  }
  stateOf(candidate.id).consecutiveRefusals = 0;
  return { ok: true, answer: answerFromOpenAI(text, canon.stream) };
}

/** Serve one client request (either wire) through the candidate list until one answers. */
async function serve(clientWire: ClientWire, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, clientWire, 400, 'invalid_request_error', 'invalid_request_error', 'the request body is not one JSON object', []);
  }
  const canon = clientWire === 'openai' ? canonFromOpenAI(body) : canonFromAnthropic(body);
  const requires = requirementsFor(canon);
  const candidates = candidatesFor(requires);
  if (candidates.length === 0) {
    return sendError(res, clientWire, 503, 'no_free_model', 'no_free_model', 'no free model in the brief meets this request', []);
  }
  const failures: FailureRow[] = [];
  let lastDetail = 'no candidate was tried';
  let lastCode: string | null = null;
  for (const candidate of candidates.slice(0, MAX_ATTEMPTS)) {
    const attempt = await tryCandidate(candidate, clientWire, body, canon);
    if (attempt.ok) {
      const rendered =
        clientWire === 'openai'
          ? answerToOpenAI(attempt.answer, candidate.id, canon.stream)
          : answerToAnthropic(attempt.answer, candidate.id, canon.stream);
      sendBody(res, 200, rendered.contentType, rendered.body, failures);
      return;
    }
    failures.push(attempt.failure);
    lastDetail = attempt.detail;
    lastCode = attempt.failure.code === '' ? null : attempt.failure.code;
    log(`prompt ${manifestIndex}: ${candidate.id} refused (${attempt.detail})`);
  }
  sendError(
    res,
    clientWire,
    502,
    clientWire === 'anthropic' ? 'api_error' : 'upstream_error',
    lastCode ?? 'all_models_refused',
    `every free model that meets this request refused; last: ${lastDetail}`,
    failures,
  );
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', 'not_found', `no route ${req.method ?? ''} ${url.pathname}`, []);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/**
 * POST /register with the port and the one catalog; the env answers with the prompt count.
 * The env may still be starting when this server is, so a failed attempt is retried for a
 * little while; the env keeps the first catalog it is given, so a retry can never replace it.
 */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 20; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(3000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }
  throw lastError instanceof Error ? lastError : new Error('register failed');
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte; wait for stdout to flush.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    const run = queue.then(() => handle(req, res));
    queue = run.catch(() => undefined);
    void run.catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'internal_error', 'the server failed to answer', []);
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's own port on one layout; the ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

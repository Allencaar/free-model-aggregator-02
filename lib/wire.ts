// Translation between the two client wires and the one upstream wire.
//
// The client speaks OpenAI Chat Completions or Anthropic Messages; the gateway speaks Chat
// Completions alone, and every free model is dialled there. So an OpenAI request rides upstream
// almost as it came, and an Anthropic request is first read into one canonical form and written
// out as Chat Completions. Every upstream answer is read back into a canonical answer — its text,
// its tool calls, its usage, its finish reason — and written out in the client's wire, in JSON or
// as an event stream. Buffering the upstream answer whole before writing anything is deliberate:
// the router may still fail over to another model, which a half-sent stream would foreclose, and
// the buffered form makes the emitted bytes a pure function of the request and the answer, which
// is what a replay compares.
//
// THE MODEL FIELD
//   A response's `model` must name the model really dialled: the gateway's record is the house's
//   witness, and a prompt whose reported model has no successful call in it earns nothing. The
//   provider's own answer names the model without the gateway's `openrouter/` prefix, so every
//   writer here stamps the dialled id — the brief's spelling — on the answer and on every chunk.
//
// Answers carry fixed ids and a fixed `created`: nothing a response holds reads the clock.

import { isRecord, type Json } from './models.ts';

export type ClientWire = 'openai' | 'anthropic';

/** One tool the client declared, in canonical form. */
export interface CanonTool {
  name: string;
  description: string;
  parameters: unknown;
}

export type CanonPart =
  | { kind: 'text'; text: string }
  | { kind: 'image'; mediaType: string | null; data: string | null; url: string | null }
  | { kind: 'tool_use'; id: string; name: string; input: unknown }
  | { kind: 'tool_result'; id: string; content: string };

export interface CanonMessage {
  role: 'user' | 'assistant' | 'tool';
  parts: CanonPart[];
}

/** A client request read into the canonical form. */
export interface CanonRequest {
  system: string;
  messages: CanonMessage[];
  tools: CanonTool[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
  topP: number | null;
  stream: boolean;
}

/** One tool call in a canonical answer. */
export interface CanonToolCall {
  id: string;
  name: string;
  /** The call's arguments as a JSON string, exactly as the provider spelled them. */
  arguments: string;
}

/** An upstream answer read into the canonical form. */
export interface CanonAnswer {
  text: string;
  toolCalls: CanonToolCall[];
  inputTokens: number | null;
  outputTokens: number | null;
  /** The upstream finish reason, in the Chat Completions vocabulary. */
  finish: string | null;
}

/** A fixed `created` for emitted answers and listings: no wall clock anywhere in a response. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for the Anthropic listing's `created_at`, for the same reason. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** Streamed text is emitted in pieces of this many characters. */
const SSE_PIECE_CHARS = 16;

// --------------------------------------------------------------------------- reading client bodies

function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** An Anthropic Messages body read into the canonical form. Its jsonMode is always false. */
export function canonFromAnthropic(body: Json): CanonRequest {
  const messages: CanonMessage[] = [];
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const entry of raw) {
    if (!isRecord(entry)) continue;
    if (entry.role !== 'user' && entry.role !== 'assistant') continue;
    const parts: CanonPart[] = [];
    const content = entry.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ kind: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            parts.push({
              kind: 'image',
              mediaType: typeof block.source.media_type === 'string' ? block.source.media_type : null,
              data: block.source.data,
              url: null,
            });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            parts.push({ kind: 'image', mediaType: null, data: null, url: block.source.url });
          }
        } else if (block.type === 'tool_use' && typeof block.name === 'string') {
          parts.push({ kind: 'tool_use', id: typeof block.id === 'string' ? block.id : '', name: block.name, input: block.input ?? {} });
        } else if (block.type === 'tool_result') {
          parts.push({
            kind: 'tool_result',
            id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
            content: typeof block.content === 'string' ? block.content : blocksText(block.content),
          });
        }
      }
    }
    messages.push({ role: entry.role, parts });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      tools.push({
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        parameters: tool.input_schema ?? { type: 'object' },
      });
    }
  }
  return {
    system: blocksText(body.system),
    messages,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stream: body.stream === true,
  };
}

/** An OpenAI Chat Completions body read into the canonical form (routing reads its demands from it). */
export function canonFromOpenAI(body: Json): CanonRequest {
  const systemParts: string[] = [];
  const messages: CanonMessage[] = [];
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const entry of raw) {
    if (!isRecord(entry)) continue;
    if (entry.role === 'system') {
      const text = blocksText(entry.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (entry.role !== 'user' && entry.role !== 'assistant' && entry.role !== 'tool') continue;
    const parts: CanonPart[] = [];
    const content = entry.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ kind: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url)) {
          const url = block.image_url.url;
          if (typeof url === 'string') {
            const data = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
            if (data !== null && data[2] !== undefined) parts.push({ kind: 'image', mediaType: data[1] ?? null, data: data[3], url: null });
            else parts.push({ kind: 'image', mediaType: null, data: null, url });
          }
        }
      }
    }
    if (Array.isArray(entry.tool_calls)) {
      for (const call of entry.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
        parts.push({
          kind: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: call.function.name,
          input: typeof call.function.arguments === 'string' ? call.function.arguments : {},
        });
      }
    }
    if (entry.role === 'tool') {
      parts.push({
        kind: 'tool_result',
        id: typeof entry.tool_call_id === 'string' ? entry.tool_call_id : '',
        content: typeof entry.content === 'string' ? entry.content : blocksText(entry.content),
      });
    }
    messages.push({ role: entry.role, parts });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
      tools.push({
        name: tool.function.name,
        description: typeof tool.function.description === 'string' ? tool.function.description : '',
        parameters: tool.function.parameters ?? { type: 'object' },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system: systemParts.join('\n'),
    messages,
    tools,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stream: body.stream === true,
  };
}

/** Whether the canonical request carries an image anywhere (a vision demand in the body itself). */
export function hasImage(canon: CanonRequest): boolean {
  return canon.messages.some((m) => m.parts.some((p) => p.kind === 'image'));
}

// --------------------------------------------------------------------------- writing the upstream body

function openAIParts(parts: CanonPart[]): { content: unknown; toolCalls: Json[]; toolMessages: Json[] } {
  const content: unknown[] = [];
  const toolCalls: Json[] = [];
  const toolMessages: Json[] = [];
  let onlyText = parts.length > 0;
  for (const part of parts) {
    if (part.kind === 'text') content.push({ type: 'text', text: part.text });
    else if (part.kind === 'image') {
      onlyText = false;
      const url = part.data !== null ? `data:${part.mediaType ?? 'image/png'};base64,${part.data}` : (part.url ?? '');
      content.push({ type: 'image_url', image_url: { url } });
    } else if (part.kind === 'tool_use') {
      onlyText = false;
      toolCalls.push({
        id: part.id,
        type: 'function',
        function: { name: part.name, arguments: typeof part.input === 'string' ? part.input : JSON.stringify(part.input ?? {}) },
      });
    } else {
      onlyText = false;
      toolMessages.push({ role: 'tool', tool_call_id: part.id, content: part.content });
    }
  }
  const single = content.length === 1 && content[0] !== undefined && (content[0] as Json).type === 'text';
  return { content: onlyText && single ? (content[0] as Json).text : content, toolCalls, toolMessages };
}

/** The canonical request written as a Chat Completions body for `model` — the gateway's one route. */
export function canonToOpenAI(canon: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  if (canon.system !== '') messages.push({ role: 'system', content: canon.system });
  for (const message of canon.messages) {
    if (message.role === 'tool') {
      for (const part of message.parts) {
        if (part.kind === 'tool_result') messages.push({ role: 'tool', tool_call_id: part.id, content: part.content });
      }
      continue;
    }
    const rendered = openAIParts(message.parts);
    const out: Json = { role: message.role, content: rendered.content };
    if (rendered.toolCalls.length > 0) out.tool_calls = rendered.toolCalls;
    messages.push(out, ...rendered.toolMessages);
  }
  const body: Json = { model, messages, stream: canon.stream };
  if (canon.tools.length > 0) {
    body.tools = canon.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (canon.maxTokens !== null) body.max_tokens = canon.maxTokens;
  if (canon.stops.length > 0) body.stop = canon.stops;
  if (canon.jsonMode) body.response_format = { type: 'json_object' };
  if (canon.temperature !== null) body.temperature = canon.temperature;
  if (canon.topP !== null) body.top_p = canon.topP;
  // A streamed answer must carry its usage in a final chunk: the client's wire check reads it.
  if (canon.stream) body.stream_options = { include_usage: true };
  return body;
}

// --------------------------------------------------------------------------- reading upstream answers

function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJson(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** A Chat Completions answer, JSON or SSE, read into the canonical answer. */
export function answerFromOpenAI(text: string, stream: boolean): CanonAnswer {
  if (!stream) {
    const body = parseJson(text);
    const choice = body !== null && Array.isArray(body.choices) && isRecord(body.choices[0]) ? body.choices[0] : null;
    const message = choice !== null && isRecord(choice.message) ? choice.message : null;
    let content = '';
    const toolCalls: CanonToolCall[] = [];
    if (message !== null) {
      if (typeof message.content === 'string') content = message.content;
      else if (Array.isArray(message.content)) {
        content = message.content
          .filter((b): b is Json => isRecord(b) && b.type === 'text' && typeof b.text === 'string')
          .map((b) => b.text as string)
          .join('');
      }
      if (Array.isArray(message.tool_calls)) {
        for (const call of message.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
          toolCalls.push({
            id: typeof call.id === 'string' && call.id !== '' ? call.id : 'call_0',
            name: call.function.name,
            arguments: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
          });
        }
      }
    }
    const usage = body !== null && isRecord(body.usage) ? body.usage : null;
    return {
      text: content,
      toolCalls,
      inputTokens: usage !== null ? numberOrNull(usage.prompt_tokens) : null,
      outputTokens: usage !== null ? numberOrNull(usage.completion_tokens) : null,
      finish: choice !== null && typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
    };
  }
  let content = '';
  const byIndex = new Map<number, { id: string; name: string; arguments: string }>();
  let finish: string | null = null;
  let input: number | null = null;
  let output: number | null = null;
  for (const frame of sseFrames(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    const chunk = parseJson(frame.data);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      input = numberOrNull(chunk.usage.prompt_tokens) ?? input;
      output = numberOrNull(chunk.usage.completion_tokens) ?? output;
    }
    if (!Array.isArray(chunk.choices) || !isRecord(chunk.choices[0])) continue;
    const choice = chunk.choices[0];
    if (isRecord(choice.delta)) {
      if (typeof choice.delta.content === 'string') content += choice.delta.content;
      if (Array.isArray(choice.delta.tool_calls)) {
        for (const call of choice.delta.tool_calls) {
          if (!isRecord(call)) continue;
          const index = typeof call.index === 'number' ? call.index : byIndex.size;
          const slot = byIndex.get(index) ?? { id: '', name: '', arguments: '' };
          if (typeof call.id === 'string' && call.id !== '') slot.id = call.id;
          if (isRecord(call.function)) {
            if (typeof call.function.name === 'string' && call.function.name !== '') slot.name = call.function.name;
            if (typeof call.function.arguments === 'string') slot.arguments += call.function.arguments;
          }
          byIndex.set(index, slot);
        }
      }
    }
    if (typeof choice.finish_reason === 'string') finish = choice.finish_reason;
  }
  const toolCalls = [...byIndex.entries()]
    .sort((a, b) => a[0] - b[0])
    .filter(([, slot]) => slot.name !== '')
    .map(([, slot]) => ({ id: slot.id !== '' ? slot.id : 'call_0', name: slot.name, arguments: slot.arguments }));
  return { text: content, toolCalls, inputTokens: input, outputTokens: output, finish };
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsObject(argumentsText: string): Json {
  if (argumentsText.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(argumentsText);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

// --------------------------------------------------------------------------- writing client answers

const STOP_TO_ANTHROPIC: Record<string, string> = {
  stop: 'end_turn',
  length: 'max_tokens',
  tool_calls: 'tool_use',
  function_call: 'tool_use',
  content_filter: 'refusal',
};

/** An upstream finish reason in the Anthropic vocabulary. */
export function stopForAnthropic(finish: string | null): string {
  return (finish !== null ? STOP_TO_ANTHROPIC[finish] : undefined) ?? 'end_turn';
}

const STOP_TO_OPENAI: Record<string, string> = {
  stop: 'stop',
  length: 'length',
  tool_calls: 'tool_calls',
  function_call: 'function_call',
  content_filter: 'content_filter',
};

/** An upstream finish reason in the OpenAI vocabulary (an unknown one is reported as `stop`). */
export function stopForOpenAI(finish: string | null): string {
  return (finish !== null ? STOP_TO_OPENAI[finish] : undefined) ?? 'stop';
}

function usageOf(answer: CanonAnswer): { input: number; output: number } {
  return { input: Math.max(0, answer.inputTokens ?? 0), output: Math.max(1, answer.outputTokens ?? 1) };
}

function splitPieces(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += SSE_PIECE_CHARS) pieces.push(text.slice(at, at + SSE_PIECE_CHARS));
  return pieces;
}

function sse(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** The canonical answer as an OpenAI completion or event stream, `model` the id really dialled. */
export function answerToOpenAI(answer: CanonAnswer, model: string, stream: boolean): { contentType: string; body: string } {
  const finish = stopForOpenAI(answer.finish);
  const usage = usageOf(answer);
  if (!stream) {
    const message: Json = { role: 'assistant', content: answer.text };
    if (answer.toolCalls.length > 0) {
      message.tool_calls = answer.toolCalls.map((call) => ({
        id: call.id,
        type: 'function',
        function: { name: call.name, arguments: call.arguments },
      }));
    }
    return {
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'chatcmpl-aggregator',
        object: 'chat.completion',
        created: FIXED_CREATED,
        model,
        choices: [{ index: 0, message, finish_reason: finish, logprobs: null }],
        usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
      }),
    };
  }
  const chunk = (delta: Json, finishReason: string | null, withUsage = false): Json => {
    const base: Json = {
      id: 'chatcmpl-aggregator',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finishReason }],
    };
    if (withUsage) {
      base.usage = { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output };
    }
    return base;
  };
  let out = sse(null, chunk({ role: 'assistant', content: '' }, null));
  for (const piece of splitPieces(answer.text)) out += sse(null, chunk({ content: piece }, null));
  answer.toolCalls.forEach((call, index) => {
    out += sse(
      null,
      chunk({ tool_calls: [{ index, id: call.id, type: 'function', function: { name: call.name, arguments: call.arguments } }] }, null),
    );
  });
  out += sse(null, chunk({}, finish));
  out += sse(null, {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  });
  out += 'data: [DONE]\n\n';
  return { contentType: 'text/event-stream', body: out };
}

/** The canonical answer as an Anthropic message or event stream, `model` the id really dialled. */
export function answerToAnthropic(answer: CanonAnswer, model: string, stream: boolean): { contentType: string; body: string } {
  const stopReason = stopForAnthropic(answer.finish);
  const usage = usageOf(answer);
  if (!stream) {
    const content: Json[] = [];
    if (answer.text !== '') content.push({ type: 'text', text: answer.text });
    for (const call of answer.toolCalls) {
      content.push({ type: 'tool_use', id: call.id, name: call.name, input: argumentsObject(call.arguments) });
    }
    if (content.length === 0) content.push({ type: 'text', text: '' });
    return {
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content,
        stop_reason: stopReason,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: usage.output },
      }),
    };
  }
  let out = sse('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_aggregator',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: usage.input, output_tokens: 0 },
    },
  });
  let index = 0;
  if (answer.text !== '') {
    out += sse('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } });
    for (const piece of splitPieces(answer.text)) {
      out += sse('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } });
    }
    out += sse('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  for (const call of answer.toolCalls) {
    out += sse('content_block_start', {
      type: 'content_block_start',
      index,
      content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} },
    });
    if (call.arguments !== '') {
      out += sse('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'input_json_delta', partial_json: call.arguments },
      });
    }
    out += sse('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  out += sse('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: stopReason, stop_sequence: null },
    usage: { output_tokens: usage.output },
  });
  out += sse('message_stop', { type: 'message_stop' });
  return { contentType: 'text/event-stream', body: out };
}

// --------------------------------------------------------------------------- errors and model lists

/** An error body in the OpenAI envelope. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, param: null, code } };
}

/** An error body in the Anthropic envelope. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** One model as GET /v1/models names it. */
export interface ListedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** GET /v1/models in the OpenAI shape: exactly the free models the catalog registered. */
export function openAIModelList(models: readonly ListedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models in the Anthropic shape: the same free model ids under a `data` list. */
export function anthropicModelList(models: readonly ListedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })),
    has_more: false,
    first_id: models.length > 0 ? models[0].id : null,
    last_id: models.length > 0 ? models[models.length - 1].id : null,
  };
}

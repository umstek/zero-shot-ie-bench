// Mock-fetch tests for the hosted provider adapters — one wire-mapping
// test plus one error path (429 or envelope failure) per adapter. No
// network: every fetchImpl is an in-memory stub.
import { describe, expect, test } from "bun:test";
import type { TrackRequestOut } from "@zsib/schema";
import { chat as fastinoChat, systemone as fastinoSystemone } from "../src/providers/fastino.ts";
import { AdapterError, getAdapter } from "../src/providers/index.ts";
import { rerank } from "../src/providers/openrouter.ts";
import { generate as typellmGenerate } from "../src/providers/typellm.ts";
import { systemone as typesafeSystemone } from "../src/providers/typesafe.ts";
import { clef as workersaiClef } from "../src/providers/workersai.ts";

type Call = {
  url: string;
  body: Record<string, unknown>;
  headers: Record<string, string>;
};

function mockFetch(reply: (call: Call, n: number) => Response): {
  fetchImpl: typeof fetch;
  calls: Call[];
} {
  const calls: Call[] = [];
  const fetchImpl = (async (url: unknown, init?: RequestInit) => {
    const call: Call = {
      url: String(url),
      body: JSON.parse((init?.body as string | undefined) ?? "null") as Record<string, unknown>,
      headers: (init?.headers ?? {}) as Record<string, string>,
    };
    calls.push(call);
    return reply(call, calls.length);
  }) as typeof fetch;
  return { fetchImpl, calls };
}

async function expectAdapterError(
  promise: Promise<unknown>,
  status: number | null,
  retryAfterMs: number | null,
  messageMatch?: RegExp,
): Promise<void> {
  try {
    await promise;
    expect.unreachable();
  } catch (err) {
    expect(err).toBeInstanceOf(AdapterError);
    const e = err as AdapterError;
    expect(e.status).toBe(status);
    expect(e.retryAfterMs).toBe(retryAfterMs);
    if (messageMatch) expect(e.message).toMatch(messageMatch);
  }
}

function req(model_id: string, items: unknown[]): TrackRequestOut {
  return { model_id, items } as unknown as TrackRequestOut;
}

function choiceItem(item_id: string, text: string, options = ["positive", "negative", "neutral"]) {
  return {
    item_id,
    input: { text, images: [] },
    question: { type: "choice", options },
  };
}

const CHOICE_ANSWER = { choice: "negative", probabilities: [0.1, 0.8, 0.1] };

describe("typesafe.systemone", () => {
  const ctx = {
    baseUrl: "https://api.typesafe.ai/v1/systemone",
    auth: { TYPESAFE_API_KEY: "ts-key" },
    endpoint: "https://api.typesafe.ai/v1/systemone",
  };

  test("posts {model, state, questions} to the catalog base_url and maps the choice row", async () => {
    const { fetchImpl, calls } = mockFetch(() =>
      Response.json({
        model: "jev-1.14.0",
        answers: { "mp-001": CHOICE_ANSWER },
        usage: { input_tokens: 5312, total_tokens: 5312 },
      }),
    );
    const res = await typesafeSystemone.send(
      req("typesafe/jev", [choiceItem("mp-001", "Worst customer service ever.")]),
      {
        ...ctx,
        fetchImpl,
      },
    );
    expect(calls).toHaveLength(1);
    expect(calls[0]?.url).toBe("https://api.typesafe.ai/v1/systemone");
    expect(calls[0]?.headers.authorization).toBe("Bearer ts-key");
    expect(calls[0]?.body.model).toBe("typesafe/jev"); // endpoint is a URL → catalog id (flagged gap)
    expect(calls[0]?.body.state).toBe("Worst customer service ever.");
    expect(calls[0]?.body.questions).toEqual({
      "mp-001": {
        type: "choice",
        instructions: "Choose the best option.",
        criteria: { positive: null, negative: null, neutral: null },
      },
    });
    expect(res.results[0]?.prediction).toBe("negative");
    expect(res.results[0]?.prob).toBeCloseTo(0.8);
    expect(res.usage).toEqual({ input_tokens: 5312, total_tokens: 5312 });
  });

  test("maps noul and score answer rows and forwards non-choice questions verbatim", async () => {
    const { fetchImpl, calls } = mockFetch(() =>
      Response.json({
        answers: { n1: { noul: 0.87 }, s1: { score: 3 } },
        usage: null,
      }),
    );
    const res = await typesafeSystemone.send(
      req("typesafe/jev", [
        {
          item_id: "n1",
          input: { text: "x" },
          question: { type: "noul", instructions: "satisfied?" },
        },
        { item_id: "s1", input: { text: "x" }, question: { type: "score", instructions: "1-5" } },
      ]),
      { ...ctx, fetchImpl },
    );
    expect(calls).toHaveLength(1); // same state → one batched request
    expect(calls[0]?.body.questions).toEqual({
      n1: { type: "noul", instructions: "satisfied?" },
      s1: { type: "score", instructions: "1-5" },
    });
    expect(res.results[0]).toMatchObject({ prediction: 0.87, prob: 0.87, error: null });
    expect(res.results[1]).toMatchObject({ prediction: 3, prob: null, error: null });
  });

  test("items with different texts ride separate requests", async () => {
    const { fetchImpl, calls } = mockFetch((_call, n) =>
      Response.json({ answers: { [`i${n}`]: CHOICE_ANSWER }, usage: { total_tokens: 3 } }),
    );
    const res = await typesafeSystemone.send(
      req("typesafe/jev", [choiceItem("i1", "first text"), choiceItem("i2", "second text")]),
      { ...ctx, fetchImpl },
    );
    expect(calls).toHaveLength(2);
    expect(calls[0]?.body.state).toBe("first text");
    expect(calls[1]?.body.state).toBe("second text");
    expect(res.results).toHaveLength(2);
    expect(res.results.every((r) => r.prediction === "negative")).toBe(true);
  });

  test("429 with Retry-After becomes AdapterError(status, retryAfterMs)", async () => {
    const { fetchImpl } = mockFetch(
      () => new Response("{}", { status: 429, headers: { "retry-after": "2" } }),
    );
    await expectAdapterError(
      typesafeSystemone.send(req("typesafe/jev", [choiceItem("mp-001", "x")]), {
        ...ctx,
        fetchImpl,
      }),
      429,
      2000,
      /typesafe HTTP 429/,
    );
  });
});

describe("workers-ai.clef", () => {
  const ctx = {
    baseUrl: "https://api.cloudflare.com/client/v4",
    auth: { CLOUDFLARE_ACCOUNT_ID: "acc123", CLOUDFLARE_AUTH_TOKEN: "cf-token" },
    endpoint: "@cf/cloudflare/clef-flash",
  };

  test("builds the /ai/run URL, unwraps the v4 envelope", async () => {
    const { fetchImpl, calls } = mockFetch(() =>
      Response.json({
        success: true,
        result: {
          model: "clef-flash",
          answers: { "mp-001": CHOICE_ANSWER },
          usage: { input_tokens: 128, total_tokens: 128 },
        },
      }),
    );
    const res = await workersaiClef.send(
      req("cloudflare/clef-flash", [choiceItem("mp-001", "Bad battery.")]),
      {
        ...ctx,
        fetchImpl,
      },
    );
    expect(calls[0]?.url).toBe(
      "https://api.cloudflare.com/client/v4/accounts/acc123/ai/run/@cf/cloudflare/clef-flash",
    );
    expect(calls[0]?.headers.authorization).toBe("Bearer cf-token");
    expect(calls[0]?.body.model).toBe("@cf/cloudflare/clef-flash"); // endpoint api_slug
    expect(calls[0]?.body.state).toBe("Bad battery.");
    expect(res.results[0]?.prediction).toBe("negative");
    expect(res.usage).toEqual({ input_tokens: 128, total_tokens: 128 });
  });

  test("a success:false envelope (HTTP 200) becomes AdapterError with the CF errors", async () => {
    const { fetchImpl } = mockFetch(() =>
      Response.json({ success: false, errors: [{ code: 7001, message: "no such model" }] }),
    );
    await expectAdapterError(
      workersaiClef.send(req("cloudflare/clef-flash", [choiceItem("mp-001", "x")]), {
        ...ctx,
        fetchImpl,
      }),
      null,
      null,
      /envelope failure.*7001/s,
    );
  });

  test("missing account id fails before any request", async () => {
    await expectAdapterError(
      workersaiClef.send(req("cloudflare/clef-flash", [choiceItem("mp-001", "x")]), {
        ...ctx,
        auth: { CLOUDFLARE_AUTH_TOKEN: "t" },
        fetchImpl: mockFetch(() => Response.json({})).fetchImpl,
      }),
      null,
      null,
      /CLOUDFLARE_ACCOUNT_ID/,
    );
  });
});

describe("fastino.systemone", () => {
  const ctx = {
    baseUrl: "https://api.fastino.ai/v1",
    auth: { FASTINO_API_KEY: "fn-key" },
    endpoint: "fastino/GLiDE",
  };

  test("posts {model, state, questions} with the api_slug to /systemone (flat payload)", async () => {
    const { fetchImpl, calls } = mockFetch(() =>
      Response.json({
        model: "fastino/GLiDE",
        answers: { "mp-001": CHOICE_ANSWER },
        usage: { input_tokens: 1944, output_tokens: 210, total_tokens: 2154 },
      }),
    );
    const res = await fastinoSystemone.send(
      req("fastino/glide", [choiceItem("mp-001", "Slow service.")]),
      {
        ...ctx,
        fetchImpl,
      },
    );
    expect(calls[0]?.url).toBe("https://api.fastino.ai/v1/systemone");
    expect(calls[0]?.headers.authorization).toBe("Bearer fn-key");
    expect(calls[0]?.body.model).toBe("fastino/GLiDE");
    expect(calls[0]?.body.state).toBe("Slow service.");
    expect(res.results[0]?.prediction).toBe("negative");
    expect(res.results[0]?.prob).toBeCloseTo(0.8);
    expect(res.usage).toEqual({ input_tokens: 1944, output_tokens: 210, total_tokens: 2154 });
  });

  test("429 honoring Retry-After becomes AdapterError(status, retryAfterMs)", async () => {
    const { fetchImpl } = mockFetch(
      () => new Response("{}", { status: 429, headers: { "retry-after": "30" } }),
    );
    await expectAdapterError(
      fastinoSystemone.send(req("fastino/glide", [choiceItem("mp-001", "x")]), {
        ...ctx,
        fetchImpl,
      }),
      429,
      30000,
      /fastino HTTP 429/,
    );
  });
});

describe("fastino.chat", () => {
  const ctx = {
    baseUrl: "https://api.fastino.ai/v1",
    auth: { FASTINO_API_KEY: "fn-key" },
    endpoint: "fastino/gliner2.5-base-v1",
  };

  test("choice: classification schema in, label + confidence out", async () => {
    const { fetchImpl, calls } = mockFetch(() =>
      Response.json({
        choices: [
          {
            message: {
              content: JSON.stringify({ classification: { label: "negative", confidence: 0.93 } }),
            },
          },
        ],
        usage: { prompt_tokens: 11, total_tokens: 11 },
      }),
    );
    const res = await fastinoChat.send(
      req("fastino/gliner2.5-base", [choiceItem("mp-001", "Broken on arrival.")]),
      {
        ...ctx,
        fetchImpl,
      },
    );
    expect(calls[0]?.url).toBe("https://api.fastino.ai/v1/chat/completions");
    expect(calls[0]?.headers.authorization).toBe("Bearer fn-key");
    expect(calls[0]?.body.model).toBe("fastino/gliner2.5-base-v1");
    expect(calls[0]?.body.messages).toEqual([{ role: "user", content: "Broken on arrival." }]);
    expect(calls[0]?.body.schema).toEqual({
      classifications: [
        {
          task: "classification",
          labels: ["positive", "negative", "neutral"],
          multi_label: false,
          top_k: 1,
        },
      ],
    });
    expect(calls[0]?.body.threshold).toBe(0.5);
    expect(res.results[0]?.prediction).toBe("negative");
    expect(res.results[0]?.prob).toBeCloseTo(0.93);
    expect(res.usage).toEqual({ prompt_tokens: 11, total_tokens: 11 });
  });

  test("ner-labels: entities schema in, sorted span triples out", async () => {
    const { fetchImpl, calls } = mockFetch(() =>
      Response.json({
        choices: [
          {
            message: {
              content: JSON.stringify({
                entities: {
                  location: [{ start: 33, end: 42, text: "Melbourne", confidence: 0.7 }],
                  person: [{ start: 0, end: 15, text: "Serena Williams", confidence: 0.98 }],
                },
              }),
            },
          },
        ],
        usage: null,
      }),
    );
    const res = await fastinoChat.send(
      req("fastino/gliner2.5-base", [
        {
          item_id: "np-001",
          input: { text: "Serena Williams won the match in Melbourne on Saturday." },
          question: { type: "ner-labels", options: ["person", "location"] },
        },
      ]),
      { ...ctx, fetchImpl },
    );
    expect(calls[0]?.body.schema).toEqual({
      entities: [
        { name: "person", description: "person" },
        { name: "location", description: "location" },
      ],
    });
    expect(res.results[0]?.prediction).toEqual([
      [0, 15, "person"],
      [33, 42, "location"],
    ]);
    expect(res.results[0]?.prob).toBeNull();
  });

  test("429 (cold start) becomes AdapterError with Retry-After", async () => {
    const { fetchImpl } = mockFetch(
      () => new Response("{}", { status: 429, headers: { "retry-after": "20" } }),
    );
    await expectAdapterError(
      fastinoChat.send(req("fastino/gliner2.5-base", [choiceItem("mp-001", "x")]), {
        ...ctx,
        fetchImpl,
      }),
      429,
      20000,
      /fastino HTTP 429/,
    );
  });
});

describe("typellm.generate", () => {
  const ctx = {
    baseUrl: "https://api.typellm.ai",
    auth: { TYPELLM_API_KEY: "tl-key" },
    endpoint: "typellm-latest",
  };

  test("choice maps to the JSON-schema enum question and result fields parse back", async () => {
    const { fetchImpl, calls } = mockFetch(() =>
      Response.json({
        id: "gen-1",
        model: "typellm-latest",
        result: { "mp-001": { answer: "negative" } },
        thinking: {},
        usage: { input_tokens: 640, thinking_tokens: 0 },
        elapsed: 0.312,
      }),
    );
    const res = await typellmGenerate.send(
      req("typellm/typellm-latest", [choiceItem("mp-001", "Arrived broken.")]),
      {
        ...ctx,
        fetchImpl,
      },
    );
    expect(calls[0]?.url).toBe("https://api.typellm.ai/v1/generate");
    expect(calls[0]?.headers.authorization).toBe("Bearer tl-key");
    expect(calls[0]?.body).toEqual({
      context: "Arrived broken.",
      questions: {
        "mp-001": {
          type: "object",
          properties: { answer: { type: "string", enum: ["positive", "negative", "neutral"] } },
        },
      },
      model: "typellm-latest",
      timeout: 60,
    });
    expect(res.results[0]?.prediction).toBe("negative");
    expect(res.results[0]?.prob).toBeNull();
    expect(res.usage).toEqual({ input_tokens: 640, thinking_tokens: 0 });
  });

  test("missing result fields and non-choice questions become item errors", async () => {
    const { fetchImpl, calls } = mockFetch(() =>
      Response.json({ result: {} as Record<string, unknown>, usage: null }),
    );
    const res = await typellmGenerate.send(
      req("typellm/typellm-latest", [
        choiceItem("mp-001", "x"),
        {
          item_id: "np-001",
          input: { text: "x" },
          question: { type: "ner-labels", options: ["person"] },
        },
      ]),
      { ...ctx, fetchImpl },
    );
    expect(calls).toHaveLength(1); // the choice item rides; the ner item never does
    expect(calls[0]?.body.questions).toHaveProperty("mp-001");
    expect(res.results[0]).toMatchObject({ prediction: null, error: "no answer for item" });
    expect(res.results[1]?.prediction).toBeNull();
    expect(res.results[1]?.error).toMatch(/choice questions only/);
  });

  test("429 with Retry-After becomes AdapterError(status, retryAfterMs)", async () => {
    const { fetchImpl } = mockFetch(
      () => new Response("{}", { status: 429, headers: { "retry-after": "5" } }),
    );
    await expectAdapterError(
      typellmGenerate.send(req("typellm/typellm-latest", [choiceItem("mp-001", "x")]), {
        ...ctx,
        fetchImpl,
      }),
      429,
      5000,
      /typellm HTTP 429/,
    );
  });
});

describe("openrouter.rerank", () => {
  const ctx = {
    baseUrl: "https://openrouter.ai/api/v1",
    auth: { OPENROUTER_API_KEY: "or-key" },
    endpoint: "cohere/rerank-4-pro",
  };

  test("choice becomes (query, documents); argmax wins with softmaxed prob", async () => {
    const { fetchImpl, calls } = mockFetch(() =>
      Response.json({
        results: [
          { index: 2, relevance_score: 0.2 },
          { index: 1, relevance_score: 0.9 },
          { index: 0, relevance_score: 0.1 },
        ],
        usage: { total_tokens: 7, search_units: 1 },
      }),
    );
    const res = await rerank.send(
      req("cohere/rerank-4-pro", [choiceItem("mp-001", "The food was terrible.")]),
      {
        ...ctx,
        fetchImpl,
      },
    );
    expect(calls[0]?.url).toBe("https://openrouter.ai/api/v1/rerank");
    expect(calls[0]?.headers.authorization).toBe("Bearer or-key");
    expect(calls[0]?.body).toEqual({
      model: "cohere/rerank-4-pro",
      query: "The food was terrible.",
      documents: ["positive", "negative", "neutral"],
    });
    expect(res.results[0]?.prediction).toBe("negative"); // index 1 scored highest
    // softmax at the max: exp(0) / (exp(0) + exp(-0.8) + exp(-0.7))
    expect(res.results[0]?.prob).toBeCloseTo(0.5139, 3);
    expect(res.usage).toEqual({ total_tokens: 7, search_units: 1 });
  });

  test("429 becomes AdapterError with status and Retry-After", async () => {
    const { fetchImpl } = mockFetch(
      () => new Response("{}", { status: 429, headers: { "retry-after": "3" } }),
    );
    await expectAdapterError(
      rerank.send(req("cohere/rerank-4-pro", [choiceItem("mp-001", "x")]), { ...ctx, fetchImpl }),
      429,
      3000,
      /openrouter HTTP 429/,
    );
  });
});

describe("adapter registry", () => {
  test("resolves every adapter id, rejects unknown ones", () => {
    for (const id of [
      "openrouter.systemone",
      "openrouter.rerank",
      "typesafe.systemone",
      "workers-ai.clef",
      "fastino.systemone",
      "fastino.chat",
      "typellm.generate",
    ]) {
      expect(getAdapter(id).id).toBe(id);
    }
    expect(() => getAdapter("nope.nothing")).toThrow();
  });
});

import test, { after, before } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { server, validChat } from "../server.mjs";
const originalFetch = globalThis.fetch;
let fakeEvents = "",
  received = [];
before(async () => {
  globalThis.fetch = async (url, options) => {
    if (String(url).endsWith("/api/tags"))
      return Response.json({ models: [{ name: "qwen3.5:4b" }] });
    received = JSON.parse(options.body).messages;
    return new Response(fakeEvents, {
      headers: { "Content-Type": "application/x-ndjson" },
    });
  };
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
});
after(async () => {
  globalThis.fetch = originalFetch;
  server.closeAllConnections();
  await new Promise((resolve) => server.close(resolve));
});
function request({
  path = "/api/chat",
  method = "POST",
  headers = {},
  chunks,
} = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        host: "127.0.0.1",
        port: server.address().port,
        path,
        method,
        headers: {
          Host: "127.0.0.1:3210",
          "Content-Type": "application/json",
          ...headers,
        },
      },
      (res) => {
        let text = "";
        res.setEncoding("utf8");
        res.on("data", (c) => (text += c));
        res.on("end", () => resolve({ status: res.statusCode, text }));
      },
    );
    req.on("error", reject);
    const parts = chunks || [
      Buffer.from(
        JSON.stringify({
          model: "qwen3.5:4b",
          messages: [{ role: "user", content: "Olá" }],
        }),
      ),
    ];
    for (const part of parts) req.write(part);
    req.end();
  });
}
test("forwards final NDJSON record even without newline and reports real metrics", async () => {
  fakeEvents =
    JSON.stringify({ message: { content: "Olá, mundo" } }) +
    "\n" +
    JSON.stringify({
      done: true,
      eval_count: 5,
      prompt_eval_count: 9,
      eval_duration: 1000000000,
      total_duration: 2000000000,
      done_reason: "stop",
    });
  const result = await request();
  const events = result.text.trim().split("\n").map(JSON.parse);
  assert.equal(result.status, 200);
  assert.equal(events.at(-1).type, "done");
  assert.equal(events.at(-1).tokensPerSecond, 5);
  assert.equal(events[1].content, "Olá, mundo");
});
test("reports a truncated stream as error", async () => {
  fakeEvents = JSON.stringify({ message: { content: "parcial" } }) + "\n";
  const result = await request();
  assert.equal(JSON.parse(result.text.trim().split("\n").at(-1)).type, "error");
});
test("stops at upstream errors", async () => {
  fakeEvents =
    JSON.stringify({ error: "sem memória" }) +
    "\n" +
    JSON.stringify({ message: { content: "should not appear" }, done: true });
  const result = await request();
  assert.ok(result.text.includes("sem memória"));
  assert.ok(!result.text.includes("should not appear"));
  assert.ok(!result.text.includes('"type":"done"'));
});
test("rejects foreign origins, bad hosts, invalid URLs, and cloud models", async () => {
  assert.equal(
    (await request({ headers: { Origin: "https://example.com" } })).status,
    403,
  );
  assert.equal(
    (await request({ headers: { Host: "example.com" } })).status,
    403,
  );
  assert.equal((await request({ path: "//[" })).status, 400);
  assert.throws(() =>
    validChat({
      model: "qwen-cloud",
      messages: [{ role: "user", content: "hello" }],
    }),
  );
  assert.throws(() =>
    validChat({
      model: "qwen3.5:4b",
      messages: [{ role: "system", content: "hello" }],
    }),
  );
});
test("preserves Portuguese when a UTF-8 code point spans request chunks", async () => {
  fakeEvents = JSON.stringify({ message: { content: "ok" }, done: true });
  const bytes = Buffer.from(
    JSON.stringify({
      model: "qwen3.5:4b",
      messages: [{ role: "user", content: "Olá" }],
    }),
  );
  const split = bytes.indexOf(Buffer.from("á")) + 1;
  const result = await request({
    chunks: [bytes.subarray(0, split), bytes.subarray(split)],
  });
  assert.equal(result.status, 200);
  assert.equal(received.at(-1).content, "Olá");
});

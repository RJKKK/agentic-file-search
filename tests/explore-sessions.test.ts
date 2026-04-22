import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  encodeSseEvent,
  ExploreSessionManager,
  type ExploreStreamEvent,
} from "../src/index.js";

describe("explore session manager", () => {
  it("records history, replays subscriptions, and serializes SSE events", async () => {
    const manager = new ExploreSessionManager();
    const session = manager.createSession({
      task: "Find the answer",
      documentIds: ["doc-1"],
      collectionId: null,
      dbPath: "storage.sqlite",
      batchMode: "force",
      batchSize: 2,
      batchThreshold: 1,
    });
    session.batchSummaries.push({ batch_index: 1, batch_answer: "Answer 1" });
    session.cumulativeAnswer = "Answer 1";

    const first = session.publish("start", { task: session.task });
    const subscription = manager.subscribe(session.sessionId);
    assert.equal(subscription.session?.sessionId, session.sessionId);
    assert.deepEqual(subscription.history?.map((event) => event.type), ["start"]);

    const second = session.publish("tool_call", { tool_name: "glob" });
    const queued = (await subscription.queue?.next()) as IteratorResult<ExploreStreamEvent>;
    assert.equal(queued.value, second);
    assert.equal(second.sequence, first.sequence + 1);

    const encoded = encodeSseEvent(session.sessionId, second);
    assert.match(encoded, /^event: tool_call\n/);
    assert.match(encoded, /"session_id":/);
    assert.match(encoded, /"tool_name":"glob"/);
    const snapshot = session.snapshot();
    assert.equal(snapshot.batch_mode, "force");
    assert.equal(snapshot.batch_size, 2);
    assert.equal(snapshot.batch_threshold, 1);
    assert.deepEqual(snapshot.batch_summaries, [{ batch_index: 1, batch_answer: "Answer 1" }]);
    assert.equal(snapshot.cumulative_answer, "Answer 1");

    manager.unsubscribe(session, subscription.queue!);
    const closed = await subscription.queue!.next();
    assert.equal(closed.done, true);
  });

  it("drops expired terminal sessions after the retention window", () => {
    const manager = new ExploreSessionManager({ retentionMinutes: 0 });
    const session = manager.createSession({
      task: "Done",
      documentIds: ["doc-1"],
    });
    session.status = "completed";
    session.updatedAt = new Date(Date.now() - 1000);

    manager.cleanup();
    assert.equal(manager.getSession(session.sessionId), null);
  });
});

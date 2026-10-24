import { describe, expect, test } from "bun:test";
import * as schema from "@zsib/schema";

describe("workspace wiring", () => {
  test("schema package is importable from bench", () => {
    expect(schema.Model).toBeDefined();
    expect(schema.RunMeta).toBeDefined();
  });
});

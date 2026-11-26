import { describe, expect, it } from "bun:test";
import {
  commentsEndpoint,
  isLocalCommentsApi,
  PROD_COMMENTS_API,
  resolveCommentsApiBase,
  STAGING_COMMENTS_API,
  threadForPathname,
} from "./comments";

describe("threadForPathname", () => {
  it("maps root pathname to overview", () => {
    expect(threadForPathname("/")).toBe("overview");
    expect(threadForPathname("")).toBe("overview");
  });

  it("maps top-level and nested model paths cleanly", () => {
    expect(threadForPathname("/picks")).toBe("picks");
    expect(threadForPathname("/picks/")).toBe("picks");
    expect(threadForPathname("/model/fastino/gliner2.5-base")).toBe("model/fastino/gliner2.5-base");
  });

  it("strips traversal segments and sanitizes invalid characters", () => {
    expect(threadForPathname("/../explore/foo bar")).toBe("explore/foo-bar");
  });
});

describe("resolveCommentsApiBase", () => {
  it("routes staging host to staging comments worker", () => {
    expect(resolveCommentsApiBase("zsib.staging.umstek.com")).toBe(STAGING_COMMENTS_API);
  });

  it("defaults to production comments worker", () => {
    expect(resolveCommentsApiBase("zsib.umstek.com")).toBe(PROD_COMMENTS_API);
    expect(resolveCommentsApiBase(undefined)).toBe(PROD_COMMENTS_API);
  });

  it("honors explicit override and strips trailing slash", () => {
    expect(resolveCommentsApiBase("zsib.umstek.com", "http://127.0.0.1:8787/")).toBe(
      "http://127.0.0.1:8787",
    );
  });
});

describe("commentsEndpoint & isLocalCommentsApi", () => {
  it("builds convention-based collection URLs", () => {
    expect(commentsEndpoint("https://comments.umstek.com/", "model/fastino/gliner2.5-base")).toBe(
      "https://comments.umstek.com/zsib/model/fastino/gliner2.5-base/comments",
    );
  });

  it("detects local dev APIs", () => {
    expect(isLocalCommentsApi("http://127.0.0.1:8787")).toBe(true);
    expect(isLocalCommentsApi("http://localhost:8787")).toBe(true);
    expect(isLocalCommentsApi("https://comments.umstek.com")).toBe(false);
  });
});

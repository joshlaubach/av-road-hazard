import { describe, expect, it } from "vitest";
import { MAX_UPLOAD_BYTES, checkFile, friendlyError } from "./client";

describe("friendlyError", () => {
  it("gives a plain reason for each status the API uses", () => {
    expect(friendlyError(413)).toContain("too big");
    expect(friendlyError(415)).toContain("JPEG and PNG");
    expect(friendlyError(422)).toContain("could not read");
    expect(friendlyError(503)).toContain("not ready");
    expect(friendlyError(0)).toContain("could not reach");
    expect(friendlyError(500)).toContain("code 500");
  });
});

describe("checkFile", () => {
  it("accepts a JPEG or PNG under the cap", () => {
    expect(checkFile({ type: "image/jpeg", size: 1000 })).toBeNull();
    expect(checkFile({ type: "image/png", size: MAX_UPLOAD_BYTES })).toBeNull();
  });
  it("refuses other types and files over the cap", () => {
    expect(checkFile({ type: "image/gif", size: 10 })).toContain("JPEG and PNG");
    expect(checkFile({ type: "image/jpeg", size: MAX_UPLOAD_BYTES + 1 })).toContain("too big");
  });
});

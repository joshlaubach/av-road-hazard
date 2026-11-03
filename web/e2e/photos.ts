import { resolve } from "node:path";

/** Committed Stellenbosch photos (provenance and licence note: tests/fixtures/stellenbosch/SOURCE.json). */
const FOLDER = resolve(__dirname, "../../tests/fixtures/stellenbosch");

export const PHOTO_ONE = resolve(FOLDER, "G0011493.jpg");
export const PHOTO_TWO = resolve(FOLDER, "G0011494.jpg");

/** A file that is not a photo, for the wrong-type error. */
export const NOT_A_PHOTO = { name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("not a photo") };

export const GAUGE_NAME = /^Hazard score \d+ out of 100: .+/;

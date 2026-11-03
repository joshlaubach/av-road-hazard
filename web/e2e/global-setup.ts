/** Runs once after both servers are up. Throws (and so stops the whole run) unless the stack is local and stubbed. */
export default async function globalSetup(): Promise<void> {
  const { assertStubStack } = await import("./stack.mjs");
  await assertStubStack();
}

import { expect, test, type Page } from "@playwright/test";
import { GAUGE_NAME } from "./photos";

const STEP_SECONDS = 4.5; // Walkthrough.tsx

async function openExample(page: Page) {
  await page.goto("/");
  await page.getByRole("button", { name: /Potholes it finds/ }).click();
  await expect(page.getByRole("img", { name: GAUGE_NAME })).toBeVisible();
}

function stepButtons(page: Page) {
  return page.getByRole("list", { name: "Steps" }).getByRole("button");
}

function stepHeading(page: Page, n: number, count: number) {
  return page.getByRole("heading", { name: new RegExp(`^Step ${n} of ${count}:`) });
}

test.describe("Example walkthrough", () => {
  test("steps through every step with Back, Next and the numbered buttons", async ({ page }) => {
    await openExample(page);
    const steps = stepButtons(page);
    const count = await steps.count();
    expect(count).toBeGreaterThan(1);
    const next = page.getByRole("button", { name: "Next", exact: true });
    const back = page.getByRole("button", { name: "Back", exact: true });

    // It opens on the finished result.
    await expect(stepHeading(page, count, count)).toBeVisible();
    await expect(next).toBeDisabled();
    await expect(back).toBeEnabled();

    // The numbered buttons reach every step.
    for (let i = 0; i < count; i++) {
      await steps.nth(i).click();
      await expect(steps.nth(i)).toHaveAttribute("aria-current", "step");
      await expect(stepHeading(page, i + 1, count)).toBeVisible();
    }

    // From the first step, Next walks forward one step at a time to the last.
    await steps.first().click();
    await expect(back).toBeDisabled();
    for (let i = 1; i < count; i++) {
      await next.click();
      await expect(stepHeading(page, i + 1, count)).toBeVisible();
    }
    await expect(next).toBeDisabled();

    // And Back walks to the first.
    for (let i = count - 1; i > 0; i--) {
      await back.click();
      await expect(stepHeading(page, i, count)).toBeVisible();
    }
    await expect(back).toBeDisabled();
  });

  test("Play moves forward by itself, Pause stops it, and the last step offers to watch again", async ({ page }) => {
    await page.clock.install();
    await openExample(page);
    const steps = stepButtons(page);
    const count = await steps.count();

    await steps.first().click(); // Play is tested after selecting an earlier step
    await page.getByRole("button", { name: "Play", exact: true }).click();
    const pause = page.getByRole("button", { name: "Pause", exact: true });
    await expect(pause).toBeVisible();

    await page.clock.runFor((STEP_SECONDS + 0.5) * 1000);
    await expect(stepHeading(page, 2, count)).toBeVisible();

    await pause.click();
    await expect(page.getByRole("button", { name: "Play", exact: true })).toBeVisible();
    await page.clock.runFor(STEP_SECONDS * 2000);
    await expect(stepHeading(page, 2, count)).toBeVisible();

    await steps.last().click();
    const again = page.getByRole("button", { name: "Watch how it was found", exact: true });
    await expect(again).toBeVisible();
    await again.click();
    await expect(stepHeading(page, 1, count)).toBeVisible();
    await expect(pause).toBeVisible();
  });

  test.describe("with reduced motion", () => {
    test.use({ reducedMotion: "reduce" });

    test("has no Play button, and Back and Next still work", async ({ page }) => {
      await openExample(page);
      const steps = stepButtons(page);
      const count = await steps.count();

      await expect(page.getByRole("button", { name: /^(Play|Pause|Watch how it was found)$/ })).toHaveCount(0);
      await steps.first().click();
      await page.getByRole("button", { name: "Next", exact: true }).click();
      await expect(stepHeading(page, 2, count)).toBeVisible();
      await page.getByRole("button", { name: "Back", exact: true }).click();
      await expect(stepHeading(page, 1, count)).toBeVisible();
    });
  });
});

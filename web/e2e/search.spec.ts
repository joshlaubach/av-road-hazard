import { expect, test, type Page } from "@playwright/test";
import { PHOTO_ONE, PHOTO_TWO } from "./photos";

async function addTwoPhotos(page: Page) {
  await page.goto("/search");
  await page.getByLabel("Choose photos").setInputFiles([PHOTO_ONE, PHOTO_TWO]);
  await expect(page.getByText("You have added 2 so far.")).toBeVisible({ timeout: 60_000 });
}

/** Click Search and wait for the API to answer, so a check never reads the previous search's results. */
async function search(page: Page) {
  await Promise.all([
    page.waitForResponse((r) => new URL(r.url()).pathname === "/api/query"),
    page.getByRole("button", { name: "Search", exact: true }).click(),
  ]);
}

// The API allows 6 uploads a minute per address (rag.uploads_per_minute), so the photos are added once
// and every search check below reuses them. The suite makes 3 upload calls per project, 6 in all.
test.describe("Search", () => {
  test("answers with numbered citations and cards, filters by object, and asks for input when empty", async ({ page }) => {
    await addTwoPhotos(page);
    const results = page.getByRole("region", { name: "Search results" });
    await page.getByLabel("Your question").fill("Which photos show a pothole?");

    await test.step("answer cites numbered photos, and each card has a thumbnail", async () => {
      await search(page);
      const answer = results.locator("div").filter({ has: page.getByRole("heading", { name: "Answer" }) });
      await expect(answer.getByRole("paragraph").first()).toContainText(/\[\d+\]/);
      await expect(results.getByRole("heading", { name: /^Result 1:/ })).toBeVisible();
      const cards = await results.getByRole("heading", { name: /^Result \d+:/ }).count();
      expect(cards).toBeGreaterThan(0);
      for (let n = 1; n <= cards; n++) {
        await expect(results.getByRole("img", { name: `Your photo, result ${n}` })).toBeVisible();
      }
    });

    const seeFilter = page.getByLabel("I can see this");
    await test.step("the object filter person keeps results", async () => {
      await seeFilter.selectOption({ label: "person" });
      await search(page);
      await expect(results.getByRole("heading", { name: /^Result 1:/ })).toBeVisible();
    });

    await test.step("the damage filter pothole keeps results", async () => {
      await page.getByLabel("there is this kind of damage").selectOption({ label: "pothole" });
      await search(page);
      await expect(results.getByRole("heading", { name: /^Result 1:/ })).toBeVisible();
      await page.getByLabel("there is this kind of damage").selectOption({ label: "any" });
    });

    await test.step("the object filter truck returns none", async () => {
      await seeFilter.selectOption({ label: "truck" });
      await search(page);
      await expect(page.getByText("No photos matched.")).toBeVisible();
      await expect(results.getByRole("heading", { name: /^Result \d+:/ })).toHaveCount(0);
    });

    await test.step("an empty search asks for a question or a photo", async () => {
      await page.getByLabel("Your question").fill("");
      await seeFilter.selectOption({ label: "anything" });
      await page.getByRole("button", { name: "Search", exact: true }).click(); // no request: the page refuses first
      await expect(
        page.getByRole("alert").filter({ hasText: "Type a question or choose a photo to search with." }),
      ).toBeVisible();
    });
  });
});

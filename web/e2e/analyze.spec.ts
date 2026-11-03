import { expect, test } from "@playwright/test";
import { GAUGE_NAME, NOT_A_PHOTO, PHOTO_ONE } from "./photos";

test.describe("Analyze a photo", () => {
  test("shows the gauge and a plain-language headline for an uploaded photo", async ({ page }) => {
    await page.goto("/");
    await page.getByLabel(/Choose a road photo/).setInputFiles(PHOTO_ONE);

    await expect(page.getByRole("img", { name: GAUGE_NAME })).toBeVisible({ timeout: 30_000 });
    const headline = page.getByRole("region", { name: "Result" }).getByRole("heading", { level: 2 });
    await expect(headline).toHaveText(/\S/);
  });

  test("rejects a file that is not a JPEG or PNG without calling the API", async ({ page }) => {
    const apiCalls: string[] = [];
    page.on("request", (request) => {
      if (new URL(request.url()).pathname.startsWith("/api/")) apiCalls.push(request.url());
    });
    await page.goto("/");
    await page.getByLabel(/Choose a road photo/).setInputFiles(NOT_A_PHOTO);

    // Next.js also keeps an empty role="alert" route announcer, so match the message itself.
    await expect(page.getByRole("alert").filter({ hasText: "I can only read JPEG and PNG photos." })).toBeVisible();
    expect(apiCalls).toEqual([]);
  });
});

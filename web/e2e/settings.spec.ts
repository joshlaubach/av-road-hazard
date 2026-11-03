import { expect, test } from "@playwright/test";
import { PHOTO_ONE } from "./photos";

test.describe("Settings", () => {
  test("keeps the choice of Claude, which the Search page then credits", async ({ page }) => {
    await page.goto("/settings");
    const local = page.getByRole("radio", { name: /Local model/ });
    const claude = page.getByRole("radio", { name: /Claude/ });
    await expect(local).toBeChecked();

    await claude.check();
    await page.reload();
    await expect(claude).toBeChecked();

    await page.getByRole("navigation", { name: "Pages" }).getByRole("link", { name: "Search" }).click();
    await page.getByLabel("Choose photos").setInputFiles(PHOTO_ONE);
    await expect(page.getByText("You have added 1 so far.")).toBeVisible({ timeout: 60_000 });
    await page.getByLabel("Your question").fill("Which photos show a pothole?");
    await page.getByRole("button", { name: "Search", exact: true }).click();

    await expect(page.getByText(/Written by claude,/)).toBeVisible();
    await page.getByRole("link", { name: "Settings page" }).click();
    await expect(page).toHaveURL(/\/settings$/);
  });
});

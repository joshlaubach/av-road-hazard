import { expect, test, type Page } from "@playwright/test";
import { GAUGE_NAME } from "./photos";

const PAGES = [
  { path: "/", tab: "Analyze a photo" },
  { path: "/search", tab: "Search" },
  { path: "/settings", tab: "Settings" },
];

async function hasNoSideScroll(page: Page): Promise<boolean> {
  return page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);
}

test.describe("Phone layout", () => {
  test.skip(({ viewport }) => viewport?.width !== 390, "the 390 px width is the phone project");

  for (const { path, tab } of PAGES) {
    test(`${path}: no sideways scrolling, all tabs visible, current tab marked`, async ({ page }) => {
      await page.goto(path);
      const nav = page.getByRole("navigation", { name: "Pages" });
      for (const { tab: name } of PAGES) await expect(nav.getByRole("link", { name })).toBeVisible();
      await expect(nav.getByRole("link", { name: tab })).toHaveAttribute("aria-current", "page");
      await expect(nav.locator('[aria-current="page"]')).toHaveCount(1);
      expect(await hasNoSideScroll(page)).toBe(true);
    });
  }

  test("/: still no sideways scrolling with an example result open", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: /Potholes it finds/ }).click();
    await expect(page.getByRole("img", { name: GAUGE_NAME })).toBeVisible();
    expect(await hasNoSideScroll(page)).toBe(true);
  });
});

test("/credits: names both photo sources and says the demo is non-commercial", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("link", { name: "Credits and licenses" }).click();
  await expect(page.getByRole("heading", { name: "Credits and licenses" })).toBeVisible();
  await expect(page.getByText("RDD2022: Road Damage Detection 2022.")).toBeVisible();
  await expect(page.getByText(/Stellenbosch University, 2015\./)).toBeVisible();
  await expect(page.getByText(/This demo is non-commercial/)).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

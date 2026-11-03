"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { href: "/", label: "Analyze a photo" },
  { href: "/search", label: "Search" },
  { href: "/settings", label: "Settings" },
];

/** The tab bar. The current page is marked with aria-current so a screen reader announces it. */
export function Nav() {
  const path = usePathname();
  return (
    <nav className="nav" aria-label="Pages">
      {TABS.map((tab) => (
        <Link key={tab.href} href={tab.href} aria-current={path === tab.href ? "page" : undefined}>
          {tab.label}
        </Link>
      ))}
    </nav>
  );
}

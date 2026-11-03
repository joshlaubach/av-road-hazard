import type { Metadata, Viewport } from "next";
import "@fontsource-variable/arimo"; // shipped with the site: the build never asks Google for a font
import "katex/dist/katex.min.css";
import "./globals.css";
import Link from "next/link";
import { Nav } from "@/components/Nav";

export const metadata: Metadata = {
  title: "Road Hazard Check",
  description:
    "Upload a road photo and see what a road-hazard pipeline finds, explained in plain language.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Nav />
        {children}
        <p className="site-links">
          <Link href="/credits">Credits and licenses</Link>
        </p>
      </body>
    </html>
  );
}

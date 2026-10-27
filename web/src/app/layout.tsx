import type { Metadata, Viewport } from "next";
import { Fraunces, IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import "katex/dist/katex.min.css";
import "./globals.css";

const display = Fraunces({ subsets: ["latin"], variable: "--f-display", display: "swap" });
const body = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--f-body",
  display: "swap",
});
const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--f-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Road Hazard Check",
  description:
    "Upload a road photo and see what a road-hazard pipeline finds, explained in plain language.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}

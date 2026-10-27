// The browser talks to /api/* on this server, which forwards to the Python API. That keeps the page
// and the API on one origin (no CORS) and keeps the API address out of the browser.
const apiUrl = process.env.API_URL ?? "http://localhost:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${apiUrl}/:path*` }];
  },
};

export default nextConfig;

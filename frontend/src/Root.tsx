import { HashRouter, Route, Routes } from "react-router-dom";

import { App } from "./App";
import { AuthPage } from "./AuthPage";
import { LandingPage } from "./LandingPage";

/**
 * HashRouter, not BrowserRouter: this is a static SPA that may end up on plain static
 * hosting with no server-side rewrite rule configured for deep links (e.g. a visitor
 * refreshing on /login would 404 without one). Hash-based routes ("/#/login") work
 * correctly on any static host with zero server configuration.
 */
export function Root() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<AuthPage />} />
        <Route path="/register" element={<AuthPage />} />
        <Route path="/console" element={<App />} />
        <Route path="*" element={<LandingPage />} />
      </Routes>
    </HashRouter>
  );
}

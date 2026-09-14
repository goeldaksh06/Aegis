import { useState } from "react";

import { App } from "./App";
import { LandingPage } from "./LandingPage";

const VIEW_STORAGE_KEY = "aegis_view";

export function Root() {
  const [view, setView] = useState<"landing" | "console">(() =>
    sessionStorage.getItem(VIEW_STORAGE_KEY) === "console" ? "console" : "landing",
  );

  function goToConsole() {
    sessionStorage.setItem(VIEW_STORAGE_KEY, "console");
    setView("console");
  }

  function goToLanding() {
    sessionStorage.setItem(VIEW_STORAGE_KEY, "landing");
    setView("landing");
  }

  if (view === "landing") {
    return <LandingPage onLaunch={goToConsole} />;
  }

  return <App onBackToOverview={goToLanding} />;
}

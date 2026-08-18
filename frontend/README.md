# Aegis Operator Console

Minimal frontend console for driving the backend chat endpoint.

## Setup

1. Copy `.env.example` to `.env` and adjust `VITE_AEGIS_API_BASE_URL` if needed.
2. Install dependencies: `npm install`.
3. Start the app: `npm run dev`.

## Notes

- The UI only sends a chat request and renders the backend response envelope.
- The backend must allow the frontend origin through CORS. The backend defaults are already set up for common local dev ports.
- The console probes `/health` to show backend reachability and status.
- Recent request history is local UI state for operator convenience and is not persisted.
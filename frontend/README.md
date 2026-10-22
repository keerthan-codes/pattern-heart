# Pattern / Heart frontend

React + Vite application. See the [project README](../README.md) for complete setup, API, dataset, evaluation, limitations, and Vercel deployment instructions.

Run `npm run dev` with Flask on port 5001. Leave `VITE_API_BASE_URL` unset to use the existing local `/api` proxy.

For Vercel, use `frontend` as Root Directory, Vite as the preset, `npm run build` as the build command, and `dist` as Output Directory. Set `VITE_API_BASE_URL=https://pattern-heart-api.onrender.com` before building. The API client also accepts a base ending in `/api` without duplicating the prefix. `vercel.json` handles React Router deep links.

Set Render's `CORS_ORIGINS` to the exact deployed frontend origin and restart/redeploy the backend. List custom domains or specific preview origins explicitly if needed. Do not use unrestricted CORS.

Checks: `node --test tests/api.test.js`, `npm run lint`, and `npm run build`.

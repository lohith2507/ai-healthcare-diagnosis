// Thin fetch wrapper around the FastAPI backend. Uses relative URLs that Vite
// proxies to http://localhost:8000 in dev (see vite.config.js).

async function request(path, options = {}) {
  const { headers, ...rest } = options;
  const res = await fetch(path, {
    ...rest,
    // Merge caller headers instead of letting them replace Content-Type.
    headers: { "Content-Type": "application/json", ...headers },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  listModels: () => request("/api/models"),
  getModel: (slug) => request(`/api/models/${slug}`),
  predict: (slug, body) =>
    request(`/api/models/${slug}/predict`, { method: "POST", body: JSON.stringify(body) }),
  aiExplanation: (slug, body) =>
    request(`/api/models/${slug}/ai-explanation`, { method: "POST", body: JSON.stringify(body) }),
  chat: (body) => request("/api/chat", { method: "POST", body: JSON.stringify(body) }),
};

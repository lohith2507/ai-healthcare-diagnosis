import { useState } from "react";
import { api } from "../api.js";

export default function AIExplanation({ slug, modelName, values }) {
  const [state, setState] = useState({ loading: false, text: "", error: null, note: null });

  const run = async () => {
    setState({ loading: true, text: "", error: null, note: null });
    try {
      const res = await api.aiExplanation(slug, { model_name: modelName, values });
      if (res.available) setState({ loading: false, text: res.text, error: null, note: null });
      else setState({ loading: false, text: "", error: null, note: res.reason });
    } catch (e) {
      setState({ loading: false, text: "", error: e.message, note: null });
    }
  };

  return (
    <div className="ai-explain">
      <button className="btn ghost" onClick={run} disabled={state.loading}>
        {state.loading ? "Writing…" : "🧠 Explain in plain English"}
      </button>
      {state.text && <div className="ai-text">{state.text}</div>}
      {state.note && <p className="hint">{state.note}</p>}
      {state.error && <p className="error-text">{state.error}</p>}
    </div>
  );
}

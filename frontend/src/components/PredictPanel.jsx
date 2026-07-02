import { useState } from "react";
import { api } from "../api.js";
import DynamicForm from "./DynamicForm.jsx";
import ResultCard from "./ResultCard.jsx";
import ContributionChart from "./ContributionChart.jsx";
import AIExplanation from "./AIExplanation.jsx";

function initialValues(detail) {
  if (detail.input_mode === "symptoms") return { symptoms: [] };
  const v = {};
  for (const spec of detail.feature_specs) v[spec.name] = spec.default;
  return v;
}

export default function PredictPanel({ detail, prettyModel }) {
  const [values, setValues] = useState(() => initialValues(detail));
  const [modelName, setModelName] = useState(detail.best_model);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState({ loading: false, error: null });

  const submit = async () => {
    if (detail.input_mode === "symptoms" && (values.symptoms || []).length === 0) {
      setStatus({ loading: false, error: "Select at least one symptom." });
      return;
    }
    setStatus({ loading: true, error: null });
    try {
      const res = await api.predict(detail.slug, { model_name: modelName, values });
      setResult(res);
      setStatus({ loading: false, error: null });
    } catch (e) {
      setStatus({ loading: false, error: e.message });
    }
  };

  return (
    <div className="predict-grid">
      <div className="card">
        <p className="notes">{detail.notes}</p>
        <DynamicForm detail={detail} values={values} onChange={setValues} />

        <div className="controls">
          <label className="model-picker">
            Model
            <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
              {detail.models.map((m) => (
                <option key={m} value={m}>
                  {(prettyModel[m] || m) + (m === detail.best_model ? " · best" : "")}
                </option>
              ))}
            </select>
          </label>
          <button className="btn primary" onClick={submit} disabled={status.loading}>
            {status.loading ? "Predicting…" : "Predict"}
          </button>
        </div>
        {status.error && <p className="error-text">{status.error}</p>}
      </div>

      <div className="card">
        {!result ? (
          <div className="placeholder">Enter details and hit Predict to see results.</div>
        ) : (
          <>
            <ResultCard result={result} />
            {result.contributions?.length > 0 && (
              <>
                <hr />
                <ContributionChart contributions={result.contributions} />
              </>
            )}
            {result.task === "binary" && (
              <AIExplanation slug={detail.slug} modelName={modelName} values={values} />
            )}
          </>
        )}
      </div>
    </div>
  );
}

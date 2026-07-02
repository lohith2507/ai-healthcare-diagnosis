import { useMemo, useState } from "react";

function SymptomSelect({ vocab, selected, onChange }) {
  const [query, setQuery] = useState("");
  const pretty = (s) => s.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
  const filtered = useMemo(() => {
    const q = query.toLowerCase();
    return vocab.filter((s) => s.replaceAll("_", " ").includes(q)).slice(0, 60);
  }, [vocab, query]);

  const toggle = (s) =>
    onChange(selected.includes(s) ? selected.filter((x) => x !== s) : [...selected, s]);

  return (
    <div className="symptom-select">
      {selected.length > 0 && (
        <div className="chips">
          {selected.map((s) => (
            <span key={s} className="chip" onClick={() => toggle(s)}>
              {pretty(s)} ✕
            </span>
          ))}
        </div>
      )}
      <input
        className="search"
        placeholder="Search symptoms…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <div className="symptom-options">
        {filtered.map((s) => (
          <label key={s} className={`option ${selected.includes(s) ? "checked" : ""}`}>
            <input type="checkbox" checked={selected.includes(s)} onChange={() => toggle(s)} />
            {pretty(s)}
          </label>
        ))}
      </div>
    </div>
  );
}

export default function DynamicForm({ detail, values, onChange }) {
  if (detail.input_mode === "symptoms") {
    return (
      <SymptomSelect
        vocab={detail.symptom_vocab}
        selected={values.symptoms || []}
        onChange={(symptoms) => onChange({ symptoms })}
      />
    );
  }

  const set = (name, v) => onChange({ ...values, [name]: v });

  return (
    <div className="form-grid">
      {detail.feature_specs.map((spec) => (
        <div className="field" key={spec.name}>
          <label>{spec.label}</label>
          {spec.kind === "numeric" ? (
            <input
              type="number"
              value={values[spec.name]}
              min={spec.minimum}
              max={spec.maximum}
              step="any"
              onChange={(e) => set(spec.name, e.target.value === "" ? "" : Number(e.target.value))}
            />
          ) : (
            <select value={values[spec.name]} onChange={(e) => set(spec.name, e.target.value)}>
              {spec.options.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
          )}
        </div>
      ))}
    </div>
  );
}

const ICONS = {
  symptoms: "🧬",
  diabetes: "🩸",
  heart: "❤️",
  cardio: "🫀",
  stroke: "🧠",
};

export default function Sidebar({ models, slug, onSelect, prettyModel }) {
  return (
    <aside className="sidebar">
      <h3 className="sidebar-title">Models</h3>
      <ul className="model-list">
        {models.map((m) => (
          <li key={m.slug}>
            <button
              className={`model-item ${slug === m.slug ? "active" : ""}`}
              onClick={() => onSelect(m.slug)}
            >
              <span className="model-icon">{ICONS[m.slug] || "🔬"}</span>
              <span className="model-meta">
                <span className="model-name">{m.label}</span>
                <span className="model-sub">
                  {m.n_samples.toLocaleString()} patients · {prettyModel[m.best_model] || m.best_model}
                </span>
              </span>
            </button>
          </li>
        ))}
      </ul>
    </aside>
  );
}

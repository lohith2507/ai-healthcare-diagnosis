const METRIC_COLS = [
  ["accuracy", "Accuracy"],
  ["f1_macro", "F1 (macro)"],
  ["precision_macro", "Precision"],
  ["recall_macro", "Recall"],
  ["roc_auc", "ROC-AUC"],
  ["top3_accuracy", "Top-3 acc"],
  ["train_time_s", "Train (s)"],
];

const fmt = (key, v) =>
  v == null ? "—" : key === "train_time_s" ? v.toFixed(2) : v.toFixed(3);

export default function PerformancePanel({ detail, prettyModel }) {
  const metrics = detail.metrics;
  const cols = METRIC_COLS.filter(([k]) => Object.values(metrics).some((m) => m[k] != null));

  return (
    <div className="card">
      <h4>Model comparison (held-out test set)</h4>
      <div className="table-wrap">
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Model</th>
              {cols.map(([, label]) => (
                <th key={label}>{label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(metrics).map(([name, m]) => (
              <tr key={name} className={name === detail.best_model ? "best-row" : ""}>
                <td>
                  {prettyModel[name] || name}
                  {name === detail.best_model && <span className="star"> ★</span>}
                </td>
                {cols.map(([k]) => (
                  <td key={k}>{fmt(k, m[k])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="figures">
        {detail.figures.confusion_matrix && (
          <figure>
            <img src={detail.figures.confusion_matrix} alt="Confusion matrix" />
            <figcaption>Confusion matrix (best model)</figcaption>
          </figure>
        )}
        {detail.figures.importance && (
          <figure>
            <img src={detail.figures.importance} alt="Feature importance" />
            <figcaption>Top feature importances</figcaption>
          </figure>
        )}
      </div>
    </div>
  );
}

export default function DataPanel({ detail }) {
  return (
    <div className="card">
      <div className="stat-row">
        <div className="stat">
          <span className="stat-value">{detail.n_samples.toLocaleString()}</span>
          <span className="stat-label">samples</span>
        </div>
        <div className="stat">
          <span className="stat-value">{detail.target_names.length}</span>
          <span className="stat-label">classes</span>
        </div>
        <div className="stat">
          <span className="stat-value">{detail.n_features_transformed}</span>
          <span className="stat-label">features</span>
        </div>
        <div className="stat">
          <span className="stat-value">{detail.task}</span>
          <span className="stat-label">task</span>
        </div>
      </div>

      <p className="notes">{detail.notes}</p>

      {detail.figures.eda && (
        <figure className="eda">
          <img src={detail.figures.eda} alt="EDA" />
          <figcaption>Class distribution and numeric feature correlations</figcaption>
        </figure>
      )}
    </div>
  );
}

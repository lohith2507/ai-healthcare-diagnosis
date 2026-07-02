import { RadialBar, RadialBarChart, PolarAngleAxis, ResponsiveContainer } from "recharts";

function Gauge({ value, positive }) {
  const pct = Math.round(value * 100);
  const color = value >= 0.5 ? "#e05c5c" : "#2fb37a";
  const data = [{ name: "risk", value: pct, fill: color }];
  return (
    <div className="gauge">
      <ResponsiveContainer width="100%" height={200}>
        <RadialBarChart
          innerRadius="70%"
          outerRadius="100%"
          data={data}
          startAngle={180}
          endAngle={0}
        >
          <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
          <RadialBar background dataKey="value" cornerRadius={12} angleAxisId={0} />
        </RadialBarChart>
      </ResponsiveContainer>
      <div className="gauge-label">
        <span className="gauge-value" style={{ color }}>
          {pct}%
        </span>
        <span className="gauge-sub">probability of {positive}</span>
      </div>
    </div>
  );
}

export default function ResultCard({ result }) {
  if (result.task === "binary") {
    const p = result.probabilities[1];
    const positive = result.positive_class || result.target_names[1];
    const elevated = p >= 0.5;
    return (
      <div className="result-card">
        <Gauge value={p} positive={positive} />
        <div className={`verdict ${elevated ? "bad" : "good"}`}>
          {elevated ? `Elevated risk of ${positive}` : `Lower risk of ${positive}`}
          <span className="verdict-sub">
            {result.target_names[0]}: {(result.probabilities[0] * 100).toFixed(1)}% ·{" "}
            {positive}: {(p * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    );
  }

  // multiclass (symptom checker)
  const top = result.top_predictions;
  const max = top[0]?.probability || 1;
  return (
    <div className="result-card">
      <h4>Top predictions</h4>
      <div className="top-list">
        {top.map((t, i) => (
          <div className="top-row" key={t.label}>
            <span className="top-rank">#{i + 1}</span>
            <span className="top-name">{t.label}</span>
            <div className="top-bar">
              <div
                className="top-bar-fill"
                style={{ width: `${(t.probability / max) * 100}%` }}
              />
            </div>
            <span className="top-pct">{(t.probability * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

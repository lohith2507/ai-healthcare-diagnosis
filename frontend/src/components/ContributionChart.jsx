import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function ContributionChart({ contributions }) {
  if (!contributions || contributions.length === 0) return null;
  const data = [...contributions]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 8)
    .reverse()
    .map((c) => ({ name: c.feature, value: Number(c.contribution.toFixed(4)) }));

  return (
    <div>
      <h4>Why this prediction?</h4>
      <ResponsiveContainer width="100%" height={Math.max(220, data.length * 38)}>
        <BarChart data={data} layout="vertical" margin={{ left: 20, right: 20 }}>
          <XAxis type="number" hide />
          <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 12 }} />
          <Tooltip formatter={(v) => v.toFixed(4)} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.value >= 0 ? "#e05c5c" : "#2fb37a"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="hint">Red pushes toward the predicted / positive class; green pushes away.</p>
    </div>
  );
}

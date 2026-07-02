import { useRef, useState } from "react";
import { api } from "../api.js";

export default function ChatPanel() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Hi! Tell me what symptoms you're experiencing — e.g. \"I've had a sore throat, cough, and mild fever since yesterday.\"",
    },
  ]);
  const [symptoms, setSymptoms] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  const pretty = (s) => s.replaceAll("_", " ");

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setMessages((m) => [...m, { role: "user", text }]);
    setInput("");
    setLoading(true);
    try {
      const res = await api.chat({ message: text, symptoms });
      setSymptoms(res.symptoms);
      const preds = res.predictions
        .map((p) => `• ${p.disease} — ${(p.probability * 100).toFixed(0)}%`)
        .join("\n");
      const chips = res.symptoms.map(pretty).join(", ");
      const body =
        (chips ? `Detected symptoms: ${chips}\n\n` : "") +
        (preds ? `Top possibilities:\n${preds}\n\n` : "") +
        res.reply;
      setMessages((m) => [...m, { role: "assistant", text: body }]);
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", text: `Sorry, an error occurred: ${e.message}` }]);
    } finally {
      setLoading(false);
      setTimeout(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
    }
  };

  const reset = () => {
    setMessages([{ role: "assistant", text: "Conversation reset. What symptoms are you experiencing?" }]);
    setSymptoms([]);
  };

  return (
    <div className="card chat">
      <div className="chat-head">
        <p className="notes">
          Describe how you feel in plain English — the assistant maps it to symptoms, runs the model,
          and explains the result. Powered by Groq.
        </p>
        <button className="btn ghost small" onClick={reset}>
          Reset
        </button>
      </div>

      <div className="chat-window">
        {messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role}`}>
            {m.text.split("\n").map((line, j) => (
              <span key={j}>
                {line}
                <br />
              </span>
            ))}
          </div>
        ))}
        {loading && <div className="bubble assistant typing">Thinking…</div>}
        <div ref={endRef} />
      </div>

      <div className="chat-input">
        <input
          value={input}
          placeholder="Describe your symptoms…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
        />
        <button className="btn primary" onClick={send} disabled={loading}>
          Send
        </button>
      </div>
    </div>
  );
}

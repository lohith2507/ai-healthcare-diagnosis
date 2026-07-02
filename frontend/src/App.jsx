import { useEffect, useState } from "react";
import { api } from "./api.js";
import Sidebar from "./components/Sidebar.jsx";
import Disclaimer from "./components/Disclaimer.jsx";
import PredictPanel from "./components/PredictPanel.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import PerformancePanel from "./components/PerformancePanel.jsx";
import DataPanel from "./components/DataPanel.jsx";

const PRETTY_MODEL = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
  xgboost: "XGBoost",
  neural_network: "Neural Network (TensorFlow)",
};

export default function App() {
  const [models, setModels] = useState([]);
  const [slug, setSlug] = useState(null);
  const [detail, setDetail] = useState(null);
  const [tab, setTab] = useState("predict");
  const [error, setError] = useState(null);

  useEffect(() => {
    api
      .listModels()
      .then((m) => {
        setModels(m);
        if (m.length) setSlug(m[0].slug);
      })
      .catch((e) => setError(`Could not reach the API. Is the backend running? (${e.message})`));
  }, []);

  useEffect(() => {
    if (!slug) return;
    setDetail(null);
    setTab("predict");
    api.getModel(slug).then(setDetail).catch((e) => setError(e.message));
  }, [slug]);

  const tabs =
    detail?.input_mode === "symptoms"
      ? ["predict", "chat", "performance", "data"]
      : ["predict", "performance", "data"];

  const tabLabels = {
    predict: "Predict",
    chat: "Symptom Chat",
    performance: "Model Performance",
    data: "Data Insights",
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="brand">
            <span className="logo">🩺</span>
            <div>
              <h1>AI Multi-Disease Diagnostic Assistant</h1>
              <p>Predict disease risk from symptoms and patient history — explainable ML + TensorFlow.</p>
            </div>
          </div>
          <a className="repo-link" href="https://console.groq.com" target="_blank" rel="noreferrer">
            React + FastAPI
          </a>
        </div>
      </header>

      <Disclaimer />

      {error && <div className="error-banner">{error}</div>}

      <main className="layout">
        <Sidebar models={models} slug={slug} onSelect={setSlug} prettyModel={PRETTY_MODEL} />

        <section className="content">
          {!detail ? (
            <div className="card loading">Loading model…</div>
          ) : (
            <>
              <div className="content-head">
                <h2>{detail.label}</h2>
                <span className="pill">
                  Best: {PRETTY_MODEL[detail.best_model] || detail.best_model}
                </span>
              </div>

              <nav className="tabs">
                {tabs.map((t) => (
                  <button
                    key={t}
                    className={`tab ${tab === t ? "active" : ""}`}
                    onClick={() => setTab(t)}
                  >
                    {tabLabels[t]}
                  </button>
                ))}
              </nav>

              <div className="panel" key={`${detail.slug}-${tab}`}>
                {tab === "predict" && <PredictPanel detail={detail} prettyModel={PRETTY_MODEL} />}
                {tab === "chat" && <ChatPanel detail={detail} />}
                {tab === "performance" && (
                  <PerformancePanel detail={detail} prettyModel={PRETTY_MODEL} />
                )}
                {tab === "data" && <DataPanel detail={detail} />}
              </div>
            </>
          )}
        </section>
      </main>

      <footer className="footer">
        Educational portfolio project · Public datasets · Not a medical device
      </footer>
    </div>
  );
}

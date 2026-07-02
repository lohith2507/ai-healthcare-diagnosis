"""AI Multi-Disease Diagnostic Assistant — Streamlit hub.

Five trained models (symptom checker + diabetes, heart, cardiovascular, stroke)
behind one UI: enter inputs, get a prediction with confidence, see which features
drove it, and inspect each model's performance and the underlying data.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import artifacts, llm  # noqa: E402
from src.config import DATASETS, FIGURES_DIR  # noqa: E402
from src.inference import build_row, predict_proba  # noqa: E402

st.set_page_config(page_title="AI Diagnostic Assistant", page_icon="🩺", layout="wide")

PRETTY_MODEL = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "neural_network": "Neural Network (TensorFlow)",
}


@st.cache_data(show_spinner=False)
def get_metadata(slug: str) -> dict:
    return artifacts.load_metadata(slug)


@st.cache_data(show_spinner=False)
def get_metrics(slug: str) -> dict:
    return artifacts.load_metrics(slug)


def trained_slugs() -> list[str]:
    return [s for s in DATASETS if artifacts.is_trained(s)]


def disclaimer() -> None:
    st.warning(
        "This is an educational machine-learning portfolio project, **not a medical "
        "device**. Predictions are statistical estimates from public datasets and must "
        "never be used for real diagnosis. Always consult a qualified clinician.",
        icon="⚠️",
    )


# --------------------------------------------------------------------------- #
# Input widgets
# --------------------------------------------------------------------------- #
def render_symptom_input(meta: dict) -> dict:
    vocab = meta["symptom_vocab"]
    pretty = {s.replace("_", " ").title(): s for s in vocab}
    chosen = st.multiselect(
        "Select the symptoms you are experiencing",
        options=sorted(pretty.keys()),
        help="Start typing to search across all symptoms.",
    )
    return {"symptoms": [pretty[c] for c in chosen]}


def render_form_input(meta: dict) -> dict:
    values: dict = {}
    specs = meta["feature_specs"]
    cols = st.columns(2)
    for i, spec in enumerate(specs):
        c = cols[i % 2]
        if spec["kind"] == "numeric":
            values[spec["name"]] = c.number_input(
                spec["label"],
                min_value=float(spec["minimum"]),
                max_value=float(spec["maximum"]),
                value=float(spec["default"]),
                help=spec.get("help"),
            )
        else:
            options = spec["options"]
            default = spec.get("default")
            index = options.index(default) if default in options else 0
            values[spec["name"]] = c.selectbox(spec["label"], options, index=index)
    return values


# --------------------------------------------------------------------------- #
# Result rendering
# --------------------------------------------------------------------------- #
def gauge(prob: float, title: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#c0392b" if prob >= 0.5 else "#27ae60"},
                "steps": [
                    {"range": [0, 33], "color": "#eafaf1"},
                    {"range": [33, 66], "color": "#fef9e7"},
                    {"range": [66, 100], "color": "#fdedec"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def render_binary_result(meta: dict, proba) -> None:
    pos = meta["positive_class"]
    p = float(proba[1])
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(gauge(p, f"Probability of {pos}"), use_container_width=True)
    with right:
        if p >= 0.5:
            st.error(f"**Elevated risk of {pos}** — model probability {p:.1%}.")
        else:
            st.success(f"**Lower risk of {pos}** — model probability {p:.1%}.")
        st.caption(f"{meta['target_names'][0]}: {proba[0]:.1%}  ·  {pos}: {p:.1%}")


def render_multiclass_result(meta: dict, proba) -> None:
    names = meta["target_names"]
    top = sorted(range(len(proba)), key=lambda i: proba[i], reverse=True)[:3]
    st.subheader("Top predictions")
    cols = st.columns(3)
    for rank, (col, idx) in enumerate(zip(cols, top)):
        col.metric(f"#{rank + 1}  {names[idx]}", f"{proba[idx]:.1%}")

    best = names[top[0]]
    info = meta.get("disease_info", {}).get(best, {})
    if info.get("description"):
        st.markdown(f"**About {best}:** {info['description']}")
    if info.get("precautions"):
        st.markdown("**Suggested precautions:** " + ", ".join(info["precautions"]) + ".")


def render_explanation(slug: str, row: pd.DataFrame):
    """Render the feature-contribution chart and return the explanation DataFrame."""
    from src.explain import local_explanation

    with st.spinner("Explaining the prediction..."):
        try:
            expl = local_explanation(slug, row)
        except Exception as exc:  # pragma: no cover - defensive
            st.info(f"Explanation unavailable for this model ({exc}).")
            return None
    plot_df = expl.sort_values("contribution")
    fig = px.bar(
        plot_df,
        x="contribution",
        y="raw_feature",
        orientation="h",
        color="contribution",
        color_continuous_scale=["#27ae60", "#f4f4f4", "#c0392b"],
        color_continuous_midpoint=0,
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                      coloraxis_showscale=False, yaxis_title="", xaxis_title="contribution to prediction")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red pushes toward the predicted/positive class; green pushes away.")
    return expl


def render_llm_explanation(meta: dict, proba, expl) -> None:
    """Ask the Groq LLM to turn the prediction + top factors into plain English."""
    if not llm.is_configured():
        st.caption("💡 Add a `GROQ_API_KEY` to `.env` to enable AI-written plain-English explanations.")
        return
    if expl is None or expl.empty:
        return

    factors = [
        {"feature": r["raw_feature"], "contribution": float(r["contribution"])}
        for _, r in expl.head(5).iterrows()
    ]
    disease = meta.get("positive_class") or meta["label"]
    with st.spinner("Writing a plain-English summary..."):
        try:
            text = llm.explain_risk(disease, float(proba[1]), factors)
        except Exception as exc:  # pragma: no cover - network/LLM errors
            st.info(f"AI explanation unavailable ({exc}).")
            return
    st.markdown("#### 🧠 AI explanation")
    st.markdown(text)


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def predict_tab(slug: str, meta: dict, metrics: dict) -> None:
    st.caption(meta["notes"])
    with st.form(f"form_{slug}"):
        inputs = (
            render_symptom_input(meta)
            if meta["input_mode"] == "symptoms"
            else render_form_input(meta)
        )
        model_name = st.selectbox(
            "Model",
            meta["models"],
            index=meta["models"].index(meta["best_model"]),
            format_func=lambda m: PRETTY_MODEL.get(m, m)
            + (" · best" if m == meta["best_model"] else ""),
        )
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if not submitted:
        return
    if meta["input_mode"] == "symptoms" and not inputs["symptoms"]:
        st.info("Select at least one symptom to get a prediction.")
        return

    row = build_row(meta, inputs)
    proba = predict_proba(slug, model_name, row)
    if meta["task"] == "binary":
        render_binary_result(meta, proba)
    else:
        render_multiclass_result(meta, proba)

    st.divider()
    st.subheader("Why this prediction?")
    expl = render_explanation(slug, row)
    if meta["task"] == "binary":
        render_llm_explanation(meta, proba, expl)


def performance_tab(slug: str, meta: dict, metrics: dict) -> None:
    rows = []
    for name, m in metrics.items():
        rows.append(
            {
                "Model": PRETTY_MODEL.get(name, name),
                "Accuracy": m.get("accuracy"),
                "F1 (macro)": m.get("f1_macro"),
                "Precision": m.get("precision_macro"),
                "Recall": m.get("recall_macro"),
                "ROC-AUC": m.get("roc_auc"),
                "Top-3 acc": m.get("top3_accuracy"),
                "Train time (s)": m.get("train_time_s"),
                "Best": "★" if name == meta["best_model"] else "",
            }
        )
    df = pd.DataFrame(rows).dropna(axis=1, how="all")
    st.dataframe(
        df.style.format({c: "{:.3f}" for c in df.columns
                         if c in {"Accuracy", "F1 (macro)", "Precision", "Recall",
                                  "ROC-AUC", "Top-3 acc"}}),
        use_container_width=True, hide_index=True,
    )

    c1, c2 = st.columns(2)
    cm = FIGURES_DIR / f"{slug}_confusion_matrix.png"
    imp = FIGURES_DIR / f"{slug}_importance.png"
    if cm.exists():
        c1.image(str(cm), caption="Confusion matrix (best model, test set)")
    if imp.exists():
        c2.image(str(imp), caption="Top feature importances")


def _symptom_predictions(slug: str, meta: dict, symptoms: list[str], k: int = 3) -> list[dict]:
    proba = predict_proba(slug, meta["best_model"], build_row(meta, {"symptoms": symptoms}))
    order = sorted(range(len(proba)), key=lambda i: proba[i], reverse=True)[:k]
    return [{"disease": meta["target_names"][i], "probability": float(proba[i])} for i in order]


def chat_tab(slug: str, meta: dict) -> None:
    st.caption(
        "Describe how you feel in plain English. The assistant maps your words to known "
        "symptoms, runs the model, and explains the results. Powered by Groq."
    )
    if not llm.is_configured():
        st.info(
            "LLM chat is not configured. Add your `GROQ_API_KEY` to a `.env` file "
            "(see `.env.example`) and restart the app.",
            icon="🔑",
        )
        return

    if st.button("🔄 Reset conversation"):
        st.session_state.pop("chat_messages", None)
        st.session_state.pop("chat_symptoms", None)

    messages = st.session_state.setdefault("chat_messages", [])
    symptoms = st.session_state.setdefault("chat_symptoms", [])

    if not messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Hi! Tell me what symptoms you're experiencing for example, "
                "*\"I've had a sore throat, cough, and mild fever since yesterday.\"*"
            )

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Describe your symptoms...")
    if not prompt:
        return

    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    vocab = meta["symptom_vocab"]
    with st.chat_message("assistant"):
        try:
            with st.spinner("Understanding your symptoms..."):
                symptoms = llm.extract_symptoms(prompt, vocab, current=symptoms)
                st.session_state["chat_symptoms"] = symptoms
            if not symptoms:
                reply = ("I couldn't map that to any known symptoms yet. Could you describe "
                         "what you're feeling in a bit more detail (e.g. fever, cough, rash)?")
                st.markdown(reply)
                messages.append({"role": "assistant", "content": reply})
                return

            preds = _symptom_predictions(slug, meta, symptoms)
            with st.spinner("Analyzing possible conditions..."):
                narration = llm.narrate_results(symptoms, preds, meta.get("disease_info", {}))

            chips = " ".join(f"`{s.replace('_', ' ')}`" for s in symptoms)
            table = "\n".join(f"- **{p['disease']}** — {p['probability']:.0%}" for p in preds)
            reply = (
                f"**Detected symptoms:** {chips}\n\n"
                f"**Top possibilities:**\n{table}\n\n{narration}"
            )
        except Exception as exc:  # pragma: no cover - network/LLM errors
            reply = f"Sorry, the assistant hit an error: {exc}"
        st.markdown(reply)
        messages.append({"role": "assistant", "content": reply})


def data_tab(slug: str, meta: dict) -> None:
    st.markdown(
        f"**Samples:** {meta['n_samples']:,}  ·  "
        f"**Task:** {meta['task']}  ·  "
        f"**Classes:** {len(meta['target_names'])}  ·  "
        f"**Transformed features:** {meta['n_features_transformed']}"
    )
    eda = FIGURES_DIR / f"{slug}_eda.png"
    if eda.exists():
        st.image(str(eda), caption="Class distribution and numeric feature correlations")


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("🩺 AI Multi-Disease Diagnostic Assistant")
    st.markdown(
        "Predict disease risk from symptoms and patient history using classical ML "
        "and TensorFlow neural networks — with transparent, explainable results."
    )

    slugs = trained_slugs()
    if not slugs:
        st.error("No trained models found. Run `python -m src.train` first.")
        return

    with st.sidebar:
        st.header("Choose a model")
        slug = st.radio(
            "Disease area",
            slugs,
            format_func=lambda s: DATASETS[s],
            label_visibility="collapsed",
        )
        st.divider()
        meta = get_metadata(slug)
        st.metric("Best model", PRETTY_MODEL.get(meta["best_model"], meta["best_model"]))
        st.caption(f"{DATASETS[slug]} · {meta['n_samples']:,} patients")

    disclaimer()
    meta = get_metadata(slug)
    metrics = get_metrics(slug)

    st.header(DATASETS[slug])
    if meta["input_mode"] == "symptoms":
        tab_predict, tab_chat, tab_perf, tab_data = st.tabs(
            ["🔮 Predict", "💬 Symptom Chat", "📊 Model performance", "🔎 Data insights"]
        )
        with tab_chat:
            chat_tab(slug, meta)
    else:
        tab_predict, tab_perf, tab_data = st.tabs(
            ["🔮 Predict", "📊 Model performance", "🔎 Data insights"]
        )
    with tab_predict:
        predict_tab(slug, meta, metrics)
    with tab_perf:
        performance_tab(slug, meta, metrics)
    with tab_data:
        data_tab(slug, meta)


if __name__ == "__main__":
    main()

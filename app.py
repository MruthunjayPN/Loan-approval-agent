"""
app.py — Gradio UI, Phase 5.

Thin wrapper over graph.py: the only pipeline import is run_application.
predict() builds a dict from raw widget values, calls run_application(),
and formats the returned AgentState into display strings -- no bounds
checks, type coercion, or business rules live here. Correct native types
(int for age/credit_score/years_employed/existing_loans, float for
annual_income/loan_amount, str for employment_type, bool for
default_history) come from the Gradio component configuration itself
(gr.Number(precision=0) converts to int -- see Gradio's own
round_to_precision), not from casting code in this file.
"""
import gradio as gr

from graph import run_application

EMPLOYMENT_TYPES = ["salaried", "self_employed", "unemployed"]


def predict(
    age,
    annual_income,
    loan_amount,
    credit_score,
    employment_type,
    years_employed,
    existing_loans,
    default_history,
):
    application = {
        "age": age,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "employment_type": employment_type,
        "years_employed": years_employed,
        "existing_loans": existing_loans,
        "default_history": default_history,
    }
    state = run_application(application)

    if state.get("validation_status") == "FAILED":
        errors = state.get("validation_errors", [])
        return (
            "**Rejected — invalid input**",
            "This application could not be evaluated because of the validation errors below.",
            "N/A",
            "VALIDATION_FAILED",
            "N/A",
            "\n".join(f"- {e}" for e in errors),
            "Not applicable — invalid input never reaches the model.",
        )

    final_decision = state["final_decision"]
    reviewer_required = state["reviewer_required"]
    probability = state.get("approval_probability")
    probability_text = f"{probability:.4f}" if probability is not None else "N/A"

    shap_top_drivers = state.get("shap_top_drivers") or []
    explanation_summary = state.get("explanation_summary")
    if explanation_summary or shap_top_drivers:
        drivers_text = "\n".join(
            f"- {d.get('feature')} ({d.get('direction')}): {d.get('contribution'):+.4f}"
            for d in shap_top_drivers
        )
        shap_text = (explanation_summary or "").strip()
        if drivers_text:
            shap_text = f"{shap_text}\n\n**Top drivers:**\n{drivers_text}" if shap_text else drivers_text
    else:
        shap_text = "SHAP explanation unavailable for this application."

    return (
        f"**{final_decision}**",
        state["decision_reason"],
        probability_text,
        ", ".join(state["triggered_rules"]),
        "Yes" if reviewer_required else "No",
        "None",
        shap_text,
    )


with gr.Blocks(title="CP07 — Loan Approval Agent") as demo:
    gr.Markdown("# Loan Approval Agent\nRuns a raw application through the full 5-agent pipeline.")

    with gr.Row():
        with gr.Column():
            age = gr.Number(label="Age", precision=0, value=35)
            annual_income = gr.Number(label="Annual Income", value=60000)
            loan_amount = gr.Number(label="Loan Amount", value=15000)
            credit_score = gr.Number(label="Credit Score", precision=0, value=700)
            employment_type = gr.Dropdown(label="Employment Type", choices=EMPLOYMENT_TYPES, value="salaried")
            years_employed = gr.Number(label="Years Employed", precision=0, value=5)
            existing_loans = gr.Number(label="Existing Loans", precision=0, value=0)
            default_history = gr.Checkbox(label="Prior Default History", value=False)
            submit = gr.Button("Evaluate Application", variant="primary")

        with gr.Column():
            decision = gr.Markdown(label="Decision")
            decision_reason = gr.Textbox(label="Decision Reason", lines=3)
            approval_probability = gr.Textbox(label="Approval Probability")
            triggered_rules = gr.Textbox(label="Triggered Rules")
            reviewer_required = gr.Textbox(label="Reviewer Required")
            validation_errors = gr.Textbox(label="Validation Errors", lines=3)
            shap_explanation = gr.Textbox(label="SHAP Explanation", lines=6)

    submit.click(
        fn=predict,
        inputs=[
            age,
            annual_income,
            loan_amount,
            credit_score,
            employment_type,
            years_employed,
            existing_loans,
            default_history,
        ],
        outputs=[
            decision,
            decision_reason,
            approval_probability,
            triggered_rules,
            reviewer_required,
            validation_errors,
            shap_explanation,
        ],
    )


if __name__ == "__main__":
    demo.launch()

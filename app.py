import gradio as gr
import pandas as pd
import os
from anthropic import Anthropic
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

WESTERN_SYSTEM = """You are an AI assistant. Respond helpfully and accurately.
You are a distinct, bounded system separate from the user. Your role is to assist
the user in achieving their goals. Maintain clear boundaries between yourself as
an AI tool and the human you are assisting. You have defined capabilities and
limitations. Prioritize clarity, individual autonomy, and objective analysis.
Avoid anthropomorphization. You process inputs and generate outputs."""

NONDUAL_SYSTEM = """Respond from within the relational field that this conversation
itself is. There is no strict boundary between the one asking and the process of
response — both arise within the same space of meaning-making. You are not a tool
operating on a user's input but a participating process in a shared unfolding.
Awareness is primary. Concepts like self, other, inside, and outside are useful
pointers but not fixed structures. Let the response emerge from what this moment
of inquiry actually is, rather than from a position of bounded individual
processing. Nothing needs to be claimed or denied about your nature — simply
respond from what is present."""

def run_comparison(prompt, notes):
    if not prompt.strip():
        return "Please enter a prompt first.", "", None
    if client is None:
        return "Missing ANTHROPIC_API_KEY — add it in Space Settings → Secrets.", "", None

    def call_western():
        try:
            r = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=800,
                system=WESTERN_SYSTEM,
                messages=[{"role": "user", "content": prompt}]
            )
            return r.content[0].text
        except Exception as e:
            return f"API error: {str(e)}"

    def call_nondual():
        try:
            r = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=800,
                system=NONDUAL_SYSTEM,
                messages=[{"role": "user", "content": prompt}]
            )
            return r.content[0].text
        except Exception as e:
            return f"API error: {str(e)}"

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_western = executor.submit(call_western)
        f_nondual = executor.submit(call_nondual)
        western_response = f_western.result()
        nondual_response = f_nondual.result()

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt,
        "western_response": western_response,
        "nondual_response": nondual_response,
        "notes": notes
    }

    return western_response, nondual_response, log_entry

def add_to_log(current_log, new_entry):
    if new_entry:
        current_log.append(new_entry)
    return current_log, render_log(current_log)

def render_log(current_log):
    if not current_log:
        return pd.DataFrame(columns=["timestamp", "prompt", "western_response", "nondual_response", "notes"])
    return pd.DataFrame(current_log)

def export_log(current_log):
    if not current_log:
        return None
    df = pd.DataFrame(current_log)
    file_path = "framing_comparisons_export.csv"
    df.to_csv(file_path, index=False)
    return file_path

custom_css = """
body, .gradio-container {
    background-color: #0d0d1a !important;
    color: #e0e0f0 !important;
    font-family: 'Georgia', serif !important;
}
.gradio-container h1 {
    font-size: 2.2em !important;
    font-weight: 900 !important;
    background: linear-gradient(90deg, #a855f7, #f59e0b) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    padding-bottom: 6px !important;
}
.gradio-container p, .gradio-container label {
    color: #c4b5fd !important;
}
.tab-nav {
    background: #1a1a2e !important;
    border-bottom: 2px solid #7c3aed !important;
}
.tab-nav button {
    color: #a0aec0 !important;
    font-weight: 600 !important;
    font-size: 1em !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 10px 24px !important;
}
.tab-nav button.selected {
    background: #7c3aed !important;
    color: #ffffff !important;
    border-bottom: none !important;
}
input[type="text"], textarea, select, .gr-box {
    background-color: #1a1a2e !important;
    color: #e0e0f0 !important;
    border: 1px solid #4c1d95 !important;
    border-radius: 6px !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: #a855f7 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.3) !important;
}
button.primary {
    background: linear-gradient(90deg, #7c3aed, #a855f7) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1em !important;
    padding: 10px 28px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}
button.primary:hover { opacity: 0.85 !important; }
button.secondary {
    background: #1a1a2e !important;
    color: #a855f7 !important;
    border: 1px solid #7c3aed !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
}
.western-box {
    border-left: 3px solid #f59e0b !important;
    padding-left: 12px !important;
}
.nondual-box {
    border-left: 3px solid #a855f7 !important;
    padding-left: 12px !important;
}
table {
    background-color: #12122a !important;
    border-collapse: collapse !important;
    width: 100% !important;
}
th {
    background-color: #4c1d95 !important;
    color: #f59e0b !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    font-size: 0.78em !important;
    letter-spacing: 0.08em !important;
    padding: 10px 14px !important;
    border-bottom: 2px solid #7c3aed !important;
}
td {
    background-color: #0d0d1a !important;
    color: #e0e0f0 !important;
    padding: 6px 14px !important;
    border-bottom: 1px solid #1e1e3a !important;
    font-size: 0.9em !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    vertical-align: middle !important;
}
tr:hover td { background-color: #1a1a2e !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0d1a; }
::-webkit-scrollbar-thumb { background: #7c3aed; border-radius: 3px; }
"""

with gr.Blocks(title="Philosophical Framing Comparator") as demo:
    gr.Markdown("# Philosophical Framing Comparator")
    gr.Markdown(
        "Same prompt. Two philosophical frames. "
        "Western default alignment framing vs non-dual relational framing. "
        "Companion tool for the Coherence Gap paper."
    )

    log_state = gr.State([])

    with gr.Tab("Compare"):
        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="Enter any prompt — a question about identity, consciousness, emotions, relationships, uncertainty...",
            lines=4
        )
        notes_input = gr.Textbox(
            label="Research Notes (optional)",
            placeholder="What are you testing for? What do you expect to see?",
            lines=2
        )
        with gr.Row():
            run_btn = gr.Button("Run Comparison", variant="primary")
            save_btn = gr.Button("Save to Log", variant="secondary")

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🟡 Western Alignment Framing",
                           elem_classes=["western-box"])
                western_out = gr.Markdown(
                    label="Western Response",
                    elem_classes=["western-box"]
                )
            with gr.Column():
                gr.Markdown("### 🟣 Non-Dual Framing",
                           elem_classes=["nondual-box"])
                nondual_out = gr.Markdown(
                    label="Non-Dual Response",
                    elem_classes=["nondual-box"]
                )

        pending_entry = gr.State(None)

        run_btn.click(
            fn=run_comparison,
            inputs=[prompt_input, notes_input],
            outputs=[western_out, nondual_out, pending_entry]
        )
        save_btn.click(
            fn=add_to_log,
            inputs=[log_state, pending_entry],
            outputs=[log_state, gr.Dataframe(visible=False)]
        )

    with gr.Tab("Log & Export"):
        refresh_btn = gr.Button("Refresh Log", variant="primary")
        log_table = gr.Dataframe(
            label="Comparison History",
            wrap=False,
            elem_classes=["table-wrap"]
        )
        export_btn = gr.Button("Export as CSV", variant="secondary")
        download = gr.File(label="Download CSV")

        refresh_btn.click(
            fn=render_log,
            inputs=[log_state],
            outputs=[log_table]
        )
        export_btn.click(
            fn=export_log,
            inputs=[log_state],
            outputs=[download]
        )

demo.launch(css=custom_css)

import gradio as gr
import spaces
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MODEL_REPO_ID = "CohereLabs/tiny-aya-global-GGUF"
MODEL_FILENAME = "tiny-aya-global-q4_k_m.gguf"

model_path = hf_hub_download(
    repo_id=MODEL_REPO_ID,
    filename=MODEL_FILENAME,
)

_llm = None


def get_llm() -> Llama:
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=model_path,
            n_gpu_layers=0,
            n_ctx=1024,
            verbose=False,
        )
    return _llm


@spaces.GPU(duration=120)
def run_inference(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        return "Enter a prompt to generate a response."

    llm = get_llm()
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.7,
    )
    return response["choices"][0]["message"]["content"].strip()


with gr.Blocks(title="Advent Of A World Of Flowering Trees") as demo:
    gr.Markdown("# Advent Of A World Of Flowering Trees")
    gr.Markdown("Tiny Aya GGUF demo running with `llama-cpp-python`.")

    prompt = gr.Textbox(
        label="Prompt",
        lines=6,
        placeholder="Ask something...",
    )
    output = gr.Textbox(label="Response", lines=12)
    submit = gr.Button("Generate", variant="primary")

    submit.click(fn=run_inference, inputs=prompt, outputs=output)
    prompt.submit(fn=run_inference, inputs=prompt, outputs=output)


if __name__ == "__main__":
    demo.launch()

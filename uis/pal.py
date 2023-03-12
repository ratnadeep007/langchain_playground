from dotenv import load_dotenv
from langchain.chains import PALChain
from langchain import OpenAI
from langchain.chains.llm import LLMChain
import gradio as gr

load_dotenv()

def setUp() -> PALChain:
    llm = OpenAI(model_name='code-davinci-002', temperature=0, max_tokens=1024)
    pal_chain = PALChain.from_math_prompt(llm, verbose=True)
    return pal_chain

def ask(question):
    chain = setUp()
    return chain.run(question)

# def reset(text_input: gr.Textbox, text_output: gr.Text):
#     text_input.update(value="", lines=1)
#     text_output.update(value="")

with gr.Blocks() as demo:
    gr.Markdown(
        """# Demo to show PAL chain
        """
    )

    question = gr.Textbox(label="Question", lines=3, placeholder="You question...")
    answer = gr.Text(label="Answer")
    with gr.Row():
        text_button = gr.Button("Sumit", variant="primary")
        # reset_button = gr.Button("Reset", variant="secondary")

    text_button.click(ask, inputs=question, outputs=answer)
    # reset_button.click(reset, inputs=[question, answer])

demo.launch()
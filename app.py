import gradio as gr
from pdfreader import read_pdf
from memory import chunk_text, retrieve_relevant_chunks, has_evidence
from model import tokenizer, model
import torch



def answer_question(context: str, question: str) -> str:
    prompt = (
        "You are answering questions strictly using the provided document text.\n"
        "Do not use any outside knowledge.\n"
        "If the answer is not explicitly stated in the document, say:\n"
        "\"The document does not provide this information.\"\n\n"
        "Document text:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = output[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()




stored_chunks = []


def load_pdf(pdf_file):
    global stored_chunks

    text = read_pdf(pdf_file.name)
    stored_chunks = chunk_text(text)

    return f"PDF loaded successfully. ({len(stored_chunks)} text chunks)"


def ask_question(question):
    if not stored_chunks:
        return "Please upload a PDF first."

    relevant_chunks = retrieve_relevant_chunks(stored_chunks, question)

    if not has_evidence(relevant_chunks, question):
        return "The document does not provide this information."

    context = "\n\n".join(relevant_chunks)
    return answer_question(context, question)




with gr.Blocks(title="PDF Question Answering (Local LLM)") as demo:
    gr.Markdown(
        """
        ##  PDF Question Answering
        Upload a PDF and ask questions.
        Answers are generated only from the document provided to system.
        """
    )

    with gr.Row():
        pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
        pdf_status = gr.Textbox(label="PDF status")

    pdf_upload.change(load_pdf, inputs=pdf_upload, outputs=pdf_status)

    question = gr.Textbox(label="Ask a question")
    answer = gr.Textbox(label="Answer", lines=4)

    ask_btn = gr.Button("Ask")
    ask_btn.click(ask_question, inputs=question, outputs=answer)

if __name__ == "__main__":
    demo.launch()

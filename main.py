from model import tokenizer, model
from pdfreader import read_pdf
from memory import chunk_text, retrieve_relevant_chunks, has_evidence
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

    # Decode only newly generated tokens
    new_tokens = output[0][len(inputs.input_ids[0]):]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return answer.strip()




if __name__ == "__main__":
    pdf_path = input("Enter PDF path: ")
    pdf_text = read_pdf(pdf_path)

    chunks = chunk_text(pdf_text)
    print(f"Loaded PDF with {len(chunks)} text chunks.\n")

    while True:
        question = input("Ask a question (or type 'exit'): ")
        if question.lower() == "exit":
            break

        relevant_chunks = retrieve_relevant_chunks(chunks, question)

        if not has_evidence(relevant_chunks, question):
            print("\nAnswer: The document does not provide this information.\n")
            continue

        context = "\n\n".join(relevant_chunks)
        answer = answer_question(context, question)
        print("\nAnswer:", answer, "\n")


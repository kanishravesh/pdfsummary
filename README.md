This project is a document-based question answering system that answers questions strictly using the content of an uploaded PDF. The system is designed to avoid hallucinations by refusing to answer questions that are not supported by the document text.

The application runs entirely locally using a DeepSeek language model and does not rely on any external APIs. A live demo of the project is hosted on Hugging Face Spaces and can be accessed at the link below.

Live demo:
https://huggingface.co/spaces/kanishravesh/pdfsumup

Overview

The goal of this project is to explore how language models can be used as controlled components in a larger system rather than as free-form chatbots. Many existing PDF question-answering tools generate confident answers even when the information is not present in the document. This project takes a stricter approach by enforcing document grounding at both the retrieval and generation stages.

If the document does not contain enough evidence to answer a question, the system explicitly responds that the information is not available. This behavior is intentional and central to the design.

How the system works

When a PDF is uploaded, its text is extracted and split into smaller chunks. These chunks act as the system’s temporary memory for that document. When a user asks a question, the system first checks whether the question has meaningful overlap with the document content. If no sufficient overlap is found, the system refuses to answer without calling the language model.

If relevant text is found, only those document chunks are passed to the model. The prompt explicitly instructs the model to use only the provided text and to avoid outside knowledge. Generation is deterministic to ensure stable and predictable behavior.

This layered approach ensures that the language model never answers questions that the document itself cannot support.

Why this project was built

This project was built to better understand the limitations of large language models and how system-level constraints can improve reliability. Rather than focusing on advanced frameworks or complex tooling, the emphasis is on clarity, control, and correctness.

The project demonstrates that preventing hallucinations is not only a prompt-writing problem but also a system design problem. By introducing explicit evidence checks and refusing unsupported questions, the overall behavior becomes more trustworthy.

Limitations::

The retrieval mechanism is intentionally simple and relies on keyword overlap rather than embeddings. This makes the system easy to understand but less accurate for large or complex documents. Performance is also limited by CPU-based inference, especially during the first startup when the model is loaded.

The system is best suited for small to medium PDFs and informational or educational content. It is not intended for production use or safety-critical applications.

Technology used

The project is implemented in Python using Hugging Face Transformers for model loading, DeepSeek Coder as the local language model, PyPDF for document parsing, and Gradio for the web interface. All inference is performed locally without external API calls.
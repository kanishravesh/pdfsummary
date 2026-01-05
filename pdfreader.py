from pypdf import PdfReader


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = []

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text.append(content)

    return "\n".join(text)

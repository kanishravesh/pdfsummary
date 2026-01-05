

def chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def retrieve_relevant_chunks(chunks: list[str], question: str, top_k: int = 3) -> list[str]:
    question_words = set(question.lower().split())
    scored = []

    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(question_words & chunk_words)
        scored.append((score, chunk))

    scored.sort(reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def has_evidence(chunks: list[str], question: str, min_overlap: int = 2) -> bool:
    question_words = set(question.lower().split())

    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        if len(question_words & chunk_words) >= min_overlap:
            return True

    return False

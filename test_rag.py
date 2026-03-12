import sys
import os

# Force utf-8 output to avoid Windows console errors
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.weaviate_rag import get_weaviate_rag_client

# ── Same 3 queries used in _load_business_context ──────────────────────────
QUERIES = [
    "What is this company called? Who is the founder? What is the product name?",
    "What does this business sell? What is the main product or service and its key features?",
    "Who are the target customers? What are the key selling points, pricing plans, and brand tone?",
]

# ── Same params now used in production (tools.py) ──────────────────────────
USE_HYBRID = False
SIMILARITY_THRESHOLD = 0.7
TOP_K = 5


def run_query(client, label: str, query: str):
    """Run one query and return (label, chunks list, output string)."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"QUERY {label}: {query}")
    print(f"  hybrid={USE_HYBRID}  threshold={SIMILARITY_THRESHOLD}  top_k={TOP_K}")
    print(sep)

    result = client.query(
        query_text=query,
        top_k=TOP_K,
        use_hybrid=USE_HYBRID,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )

    lines = []
    chunks_text = []
    if result.get("success"):
        chunks = result.get("results", [])
        distances = result.get("distances", [])
        lines.append(f"Chunks returned : {len(chunks)}")
        if distances:
            lines.append(
                f"Distances       : min={min(distances):.4f}  "
                f"max={max(distances):.4f}  "
                f"avg={sum(distances)/len(distances):.4f}"
            )
        for i, chunk in enumerate(chunks):
            doc   = chunk.get("document", "")
            score = chunk.get("score", 0)
            dist  = round(1 - score, 4)
            lines.append(f"\n[CHUNK {i+1}]  score={score:.4f}  distance={dist:.4f}")
            lines.append(doc)
            chunks_text.append(doc.strip())
    else:
        lines.append(f"FAILED: {result.get('error')}")

    output = "\n".join(lines)
    print(output)
    return label, chunks_text, output


def main():
    client = get_weaviate_rag_client()
    print(f"\n{'#'*60}")
    print(f"# RAG 3-QUERY BUSINESS CONTEXT TEST")
    print(f"# hybrid={USE_HYBRID}  threshold={SIMILARITY_THRESHOLD}  top_k={TOP_K}")
    print(f"{'#'*60}")

    all_results = []
    seen: set = set()
    merged_chunks: list = []

    for i, query in enumerate(QUERIES, 1):
        label, chunk_texts, output = run_query(client, str(i), query)
        all_results.append((label, query, output))
        for ct in chunk_texts:
            if ct and ct not in seen:
                seen.add(ct)
                merged_chunks.append(ct)

    # Print the merged deduplicated text (what the LLM will actually see)
    print(f"\n{'='*60}")
    print(f"MERGED UNIQUE CHUNKS ({len(merged_chunks)} total) — what LLM sees:")
    print(f"{'='*60}")
    merged_text = "\n\n".join(merged_chunks)
    print(merged_text)

    # Save to file
    with open("rag_test_output.txt", "w", encoding="utf-8") as f:
        f.write(f"hybrid={USE_HYBRID}  threshold={SIMILARITY_THRESHOLD}  top_k={TOP_K}\n\n")
        for label, query, output in all_results:
            f.write(f"\n{'='*60}\nQUERY {label}: {query}\n{'='*60}\n{output}\n")
        f.write(f"\n{'='*60}\nMERGED UNIQUE CHUNKS ({len(merged_chunks)} total)\n{'='*60}\n")
        f.write(merged_text + "\n")

    print(f"\n\n✅ Full output saved to rag_test_output.txt")


if __name__ == "__main__":
    main()

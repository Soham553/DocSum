from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def map_summary_to_paragraphs(paragraphs, summary_sentences):
    # Convert to embeddings
    para_texts = [p["text"] for p in paragraphs]
    para_embeddings = model.encode(para_texts, convert_to_tensor=True)
    summary_embeddings = model.encode(summary_sentences, convert_to_tensor=True)

    # Find most relevant paragraph for each summary sentence
    mapping = []
    for idx, s_emb in enumerate(summary_embeddings):
        cos_scores = util.cos_sim(s_emb, para_embeddings)[0]
        best_match_idx = int(cos_scores.argmax())
        mapping.append({
            "summary_sentence": summary_sentences[idx],
            "reference_paragraph_id": paragraphs[best_match_idx]["id"],
            "reference_text": paragraphs[best_match_idx]["text"]
        })
    return mapping

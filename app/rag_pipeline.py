
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
import os
import re

# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_DIR / "chroma_db"

DISTANCE_THRESHOLD = 1.2
FALLBACK_ANSWER = "The uploaded PDFs do not contain this information."

GROQ_MODEL = "openai/gpt-oss-20b"

_embedding_model = None
_groq_client = None


# -----------------------------------------
# LOAD MODELS
# -----------------------------------------

def get_embedding_model():

    global _embedding_model

    if _embedding_model is None:

        print("Loading embedding model...")

        _embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    return _embedding_model


def get_groq_client():

    global _groq_client

    if _groq_client is None:

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing from .env"
            )

        _groq_client = Groq(api_key=api_key)

    return _groq_client


# -----------------------------------------
# GET CHROMADB COLLECTION
# -----------------------------------------

def get_collection(collection_name):

    if not isinstance(collection_name, str) or not collection_name:
        raise ValueError("Invalid ChromaDB collection name.")

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    try:

        return client.get_collection(
            name=collection_name
        )

    except Exception as e:

        print("Collection retrieval error:", e)

        return None


# -----------------------------------------
# DEBUG COLLECTION
# -----------------------------------------

def debug_collection(collection):

    print("\n========== COLLECTION DEBUG ==========")

    print("Collection:", collection.name)
    print("Total chunks:", collection.count())

    if collection.count() == 0:

        print("Collection is empty.")

        return

    sample = collection.get(
        limit=5,
        include=["documents", "metadatas"]
    )

    for document, metadata in zip(
            sample["documents"],
            sample["metadatas"]
    ):

        print("\nSource:", metadata.get("source"))
        print("Page:", metadata.get("page"))
        print("Text:", document[:300])
        print("----------------------------")

    print("======================================\n")


# -----------------------------------------
# CONVERSATION HISTORY
# -----------------------------------------

def format_conversation_history(history):

    if not history:
        return ""

    formatted = []

    for message in history[-6:]:

        if isinstance(message, dict):

            role = message.get("role", "")
            content = message.get("content", "")

            if content:

                formatted.append(
                    f"{role}: {content}"
                )

        elif isinstance(message, str):

            if message.strip():
                formatted.append(message)

    return "\n".join(formatted)


# -----------------------------------------
# REWRITE QUESTION
# -----------------------------------------

def rewrite_question(question, conversation_history=None):

    history_text = format_conversation_history(
        conversation_history
    )

    prompt = f"""
You rewrite questions for a PDF question-answering system.

Rewrite the user's question into a clear, complete question.

Rules:
1. Correct spelling and grammar.
2. Understand informal language and short questions.
3. Expand incomplete questions.
4. Resolve references using conversation history.
5. Preserve the user's original meaning.
6. Do not answer the question.
7. Do not invent facts.
8. Return only the rewritten question.

Conversation history:
{history_text}

User question:
{question}

Rewritten question:
"""

    try:

        response = get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        rewritten = response.choices[0].message.content

        if not rewritten or not rewritten.strip():
            return question

        rewritten = rewritten.strip()

        print("\nOriginal question:", question)
        print("Rewritten question:", rewritten)

        return rewritten

    except Exception as e:

        print("Question rewriting error:", e)

        return question


# -----------------------------------------
# GENERATE SEARCH QUERIES
# -----------------------------------------

def generate_search_queries(question):

    prompt = f"""
Generate exactly 3 alternative search queries for finding
information in PDF documents.

Rules:
1. Preserve the meaning of the original question.
2. Use natural alternative phrasings.
3. Include important names and keywords.
4. Correct informal wording.
5. Use useful synonyms.
6. Do not answer the question.

Original question:
{question}

Return only 3 queries, one per line.
"""

    try:

        response = get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        output = response.choices[0].message.content

        if not output:
            return [question]

        queries = [question]

        for line in output.splitlines():

            line = re.sub(
                r"^\s*[-*•\d.)]+\s*",
                "",
                line
            ).strip()

            if (
                    line
                    and line.lower() not in [
                q.lower() for q in queries
            ]
            ):

                queries.append(line)

        queries = queries[:4]

        print("\nSearch queries:")

        for query in queries:
            print("-", query)

        return queries

    except Exception as e:

        print("Search query generation error:", e)

        return [question]


# -----------------------------------------
# KEYWORD MATCHING
# -----------------------------------------

def calculate_keyword_score(question, document):

    question = question.lower()
    document = document.lower()

    stop_words = {
        "what", "where", "when", "who", "why", "how",
        "is", "was", "are", "were", "the", "a", "an",
        "of", "in", "on", "for", "to", "did", "does",
        "do", "his", "her", "he", "she", "and",
        "s", "tell", "me", "about", "please"
    }

    question_words = set(
        re.findall(r"\b[a-z0-9]+\b", question)
    ) - stop_words

    expansions = {

        "hometown": {
            "born", "birthplace", "native", "city",
            "town", "suburb", "birth", "location"
        },

        "birthplace": {
            "born", "hometown", "native", "city",
            "town", "suburb", "birth", "location"
        },

        "born": {
            "birthplace", "hometown", "native",
            "birth", "city", "town"
        },

        "achievements": {
            "records", "milestones", "awards",
            "career", "won", "scored"
        },

        "achievement": {
            "records", "milestones", "awards",
            "career", "won", "scored"
        },

        "score": {
            "runs", "record", "highest", "best"
        },

        "best": {
            "highest", "record", "maximum"
        },

        "career": {
            "debut", "achievements", "milestones"
        }
    }

    expanded_words = set(question_words)

    for word in question_words:

        expanded_words.update(
            expansions.get(word, set())
        )

    document_words = set(
        re.findall(r"\b[a-z0-9]+\b", document)
    )

    matched_words = expanded_words.intersection(
        document_words
    )

    original_matches = question_words.intersection(
        document_words
    )

    score = len(matched_words) + (
            2 * len(original_matches)
    )

    return score


# -----------------------------------------
# RETRIEVE CONTEXT
# -----------------------------------------

def retrieve_context(question, collection_name, top_k=6):

    collection = get_collection(collection_name)

    if collection is None:

        print("No collection found.")

        return []

    debug_collection(collection)

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    model = get_embedding_model()

    search_queries = generate_search_queries(question)

    all_results = {}

    # -----------------------------------------
    # SEMANTIC SEARCH
    # -----------------------------------------

    for search_query in search_queries:

        print("\nSearching:", search_query)

        embedding = model.encode(
            search_query
        ).tolist()

        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(total_chunks, top_k * 3),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for document, metadata, distance in zip(
                documents,
                metadatas,
                distances
        ):

            print(
                f"\nRetrieved: {metadata.get('source')} | "
                f"Page: {metadata.get('page')} | "
                f"Distance: {distance:.4f}"
            )

            print("Text:", document[:200])
            print("----------------------------")

            key = (
                metadata.get("source"),
                metadata.get("page"),
                document
            )

            if (
                    key not in all_results
                    or distance < all_results[key]["distance"]
            ):

                all_results[key] = {
                    "text": document,
                    "metadata": metadata,
                    "distance": distance
                }

    # -----------------------------------------
    # KEYWORD SEARCH
    # -----------------------------------------

    print("\n========== KEYWORD SEARCH ==========")

    stored_data = collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    keyword_results = []

    for document, metadata in zip(
            stored_data["documents"],
            stored_data["metadatas"]
    ):

        keyword_score = calculate_keyword_score(
            question,
            document
        )

        print(
            f"Keyword score: {keyword_score} | "
            f"Page: {metadata.get('page')} | "
            f"Source: {metadata.get('source')}"
        )

        if keyword_score > 0:

            key = (
                metadata.get("source"),
                metadata.get("page"),
                document
            )

            semantic_distance = all_results.get(
                key,
                {}
            ).get("distance", 999)

            keyword_results.append({
                "text": document,
                "metadata": metadata,
                "keyword_score": keyword_score,
                "distance": semantic_distance
            })

    # -----------------------------------------
    # COMBINE AND RANK RESULTS
    # -----------------------------------------

    if keyword_results:

        keyword_results.sort(
            key=lambda item: (
                -item["keyword_score"],
                item["distance"]
            )
        )

        ranked_results = keyword_results

    else:

        ranked_results = sorted(
            all_results.values(),
            key=lambda item: item["distance"]
        )

        ranked_results = [
            item for item in ranked_results
            if item["distance"] <= DISTANCE_THRESHOLD
        ]

    # -----------------------------------------
    # SELECT FINAL CHUNKS
    # -----------------------------------------

    final_chunks = []

    source_counts = {}

    seen_chunks = set()

    for item in ranked_results:

        metadata = item["metadata"]

        source = metadata.get("source")

        key = (
            source,
            metadata.get("page"),
            item["text"]
        )

        if key in seen_chunks:
            continue

        seen_chunks.add(key)

        source_counts[source] = source_counts.get(
            source, 0
        )

        if source_counts[source] >= 3:
            continue

        source_counts[source] += 1

        final_chunks.append({
            "text": item["text"],
            "metadata": metadata,
            "distance": item["distance"]
        })

        if len(final_chunks) >= top_k:
            break

    print("\n========== FINAL RETRIEVAL ==========")

    print("Accepted chunks:", len(final_chunks))

    for chunk in final_chunks:

        print(
            "Accepted:",
            chunk["metadata"].get("source"),
            "| Page:",
            chunk["metadata"].get("page"),
            "| Distance:",
            chunk["distance"]
        )

    print("=====================================")

    return final_chunks


# -----------------------------------------
# GENERATE ANSWER
# -----------------------------------------

def generate_answer(question, retrieved_chunks):

    if not retrieved_chunks:
        return FALLBACK_ANSWER, []

    context_parts = []

    chunk_map = {}

    for index, chunk in enumerate(
            retrieved_chunks,
            start=1
    ):

        chunk_id = f"C{index}"

        metadata = chunk["metadata"]

        context_parts.append(
            f"""
[{chunk_id}]
Source: {metadata.get("source")}
Page: {metadata.get("page")}

Content:
{chunk["text"]}
"""
        )

        chunk_map[chunk_id] = {
            "source": metadata.get("source"),
            "page": metadata.get("page")
        }

    context = "\n".join(context_parts)

    prompt = f"""
You are a reliable PDF question-answering assistant.

Answer using ONLY the provided PDF context.

Rules:
1. Never use outside knowledge.
2. Never invent information.
3. If the answer is absent, say exactly:
   "{FALLBACK_ANSWER}"
4. Give a direct and clear answer.
5. Cite the chunk IDs that support your answer.
6. The final line must use this format:
   SOURCES: C1, C2
7. Only cite chunks that actually support the answer.
8. Do not cite unrelated chunks.

PDF CONTEXT:

{context}

QUESTION:
{question}

ANSWER:
"""

    try:

        response = get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        answer_text = response.choices[0].message.content

        if not answer_text or not answer_text.strip():
            return FALLBACK_ANSWER, []

        answer_text = answer_text.strip()

        print("\nGroq response:", answer_text)

        # Do not show references for fallback answers.
        if FALLBACK_ANSWER.casefold() in answer_text.casefold():
            return FALLBACK_ANSWER, []

        source_match = re.search(
            r"(?im)^\s*SOURCES:\s*(.*?)\s*$",
            answer_text
        )

        answer = re.sub(
            r"(?im)^\s*SOURCES:\s*.*$",
            "",
            answer_text
        ).strip()

        references = []

        if source_match:

            source_ids = re.findall(
                r"C\d+",
                source_match.group(1)
            )

            for source_id in source_ids:

                if source_id in chunk_map:

                    reference = chunk_map[source_id]

                    if reference not in references:
                        references.append(reference)

        if not answer:
            return FALLBACK_ANSWER, []

        return answer, references

    except Exception as e:

        print("Answer generation error:", e)

        return (
            "Sorry, something went wrong while generating the answer.",
            []
        )


# -----------------------------------------
# MAIN RAG FUNCTION
# -----------------------------------------

def answer_question(
        question,
        collection_name,
        conversation_history=None
):

    if not isinstance(question, str) or not question.strip():
        return FALLBACK_ANSWER, []

    if not isinstance(collection_name, str):
        raise ValueError(
            "Collection name must be a string. "
            "Check answer_question() arguments."
        )

    rewritten_question = rewrite_question(
        question,
        conversation_history
    )

    print("\nSearch question:", rewritten_question)
    print("Using collection:", collection_name)

    retrieved_chunks = retrieve_context(
        rewritten_question,
        collection_name
    )

    answer, references = generate_answer(
        rewritten_question,
        retrieved_chunks
    )

    return answer, references

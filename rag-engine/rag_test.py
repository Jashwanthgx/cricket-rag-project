import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import uuid
import re
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

QDRANT_URL     = "https://922f9913-1f6e-4d54-b394-bd41a95170ca.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.32lYkMgpft2UmFYiTsizXKRzORc0iu4GvJGl15rzQMw"
COLLECTION     = "cricket_pro"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
model  = SentenceTransformer("all-MiniLM-L6-v2")

if not os.path.exists("cricket_rag_data.jsonl"):
    print("Error: 'cricket_rag_data.jsonl' not found!")
    exit()

documents = []
payloads  = []

with open("cricket_rag_data.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        documents.append(data["text"])
        payloads.append({
            "text":            data["text"],
            "team1":           data.get("team1", ""),
            "team2":           data.get("team2", ""),
            "date":            data.get("date", ""),
            "venue":           data.get("venue", ""),
            "winner":          data.get("winner", ""),
            "mom":             data.get("mom", ""),
            "top_scorer":      data.get("top_scorer", "N/A"),
            "top_scorer_runs": data.get("top_scorer_runs", 0),
            "top_bowler":      data.get("top_bowler", "N/A")
        })

print(f"Loaded {len(documents)} summaries.")

embeddings = model.encode(documents, show_progress_bar=True)

try:
    print("Refreshing collection to update data structure...")
    client.delete_collection(COLLECTION)
except Exception:
    pass

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
)

points = [
    PointStruct(id=str(uuid.uuid4()), vector=embeddings[i].tolist(), payload=payloads[i])
    for i in range(len(documents))
]

batch_size = 50
for start in range(0, len(points), batch_size):
    client.upsert(collection_name=COLLECTION, points=points[start:start+batch_size])
print(f"All {len(documents)} matches successfully stored in Qdrant.")

AGGREGATE_PATTERNS = [
    r"\bhow many\b", r"\btotal\b", r"\bcount\b", r"\blist all\b",
    r"\bmost\b", r"\bhighest\b", r"\bbest\b", r"\baverage\b",
    r"\bevery\b", r"\bwho won\b", r"\bsummarize\b"
]

CRICKET_KEYWORDS = [
    r"\bcricket\b", r"\binnings\b", r"\bwickets?\b", r"\bbatsmen?\b",
    r"\bbowlers?\b", r"\bcentur(y|ies)\b", r"\bover\b", r"\bodi\b", r"\bt20\b",
    r"\bipl\b", r"\blbw\b", r"\bhat-trick\b", r"\bmaiden\b",
    r"\bstrike rate\b", r"\bplayer of the match\b", r"\brunscored\b",
    r"\bsix(es)?\b", r"\bboundar(y|ies)\b", r"\brun chase\b",
]

GENERAL_COMPUTING_PATTERNS = [
    r"\bpowershell\b", r"\bcmd\.exe\b", r"\bcommand prompt\b",
    r"\bset-location\b", r"\b\.exe\b", r"\bfile explorer\b",
    r"\bexecutable\b", r"\bnosql\b", r"\bpath.*spaces\b",
]

def is_aggregate_query(q: str) -> bool:
    q_lower = q.lower()
    if any(re.search(p, q_lower) for p in AGGREGATE_PATTERNS):
        return True
    
    cricket_terms = ["runs", "wickets", "match", "played", "score", "century", "mom"]
    words = q.split()
    has_potential_name = any(word[0].isupper() for word in words[1:] if len(word) > 0)
    has_cricket_intent = any(term in q_lower for term in cricket_terms)
    
    return has_potential_name and has_cricket_intent

def is_cricket_query(q: str) -> bool:
    """Return True if the query is about cricket; False for general computing or other topics."""
    q_lower = q.lower()
    has_cricket_keyword = any(re.search(p, q_lower) for p in CRICKET_KEYWORDS)
    has_general_computing = any(re.search(p, q_lower) for p in GENERAL_COMPUTING_PATTERNS)
    if has_general_computing and not has_cricket_keyword:
        return False
    if has_cricket_keyword:
        return True
    return is_aggregate_query(q)

print("\n" + "="*50)
print("Cricket RAG Assistant  | type 'quit' to exit")
print("="*50)

while True:
    query = input("\nAsk something about the matches: ").strip()
    if query.lower() in ("quit", "exit", "q"):
        break
    if not query:
        continue

    if not is_cricket_query(query):
        print("[General Query] - Answering from general knowledge using Llama 3...")
        prompt = f"""You are a helpful general-purpose assistant with expertise in computing, \
programming, and operating systems.

Answer the following question clearly and accurately. Keep these guidelines in mind:
- For questions about running executables or files (e.g. .exe files), remind the user to \
use ./ or a full absolute path and to wrap paths containing spaces in quotes.
- For PowerShell or Command Prompt questions, explain common errors such as using 'cd' on \
a file instead of a directory, or omitting the path prefix, and provide step-by-step \
instructions. Where applicable, also mention the Windows File Explorer double-click \
alternative.
- For database or programming questions, provide clear examples and explanations.

Question: {query}
Answer:"""

        response = ollama.generate(
            model="llama3",
            prompt=prompt,
            options={
                "temperature": 0.2,
                "num_predict": 800,
            },
        )
        answer = response["response"].strip()

        print("\n" + "="*40)
        print("ANSWER:")
        print(answer)
        print("="*40)
        continue

    query_vector = model.encode(query).tolist()

    if is_aggregate_query(query):
        priority_results = client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=5
        ).points
        
        all_results = client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=len(documents)
        ).points

        print(f"[Aggregate Query] - Scanning all {len(all_results)} records using Llama 3...")
        
        context_text = "PRIORITY MATCH RECORDS (FULL DETAIL):\n"
        for i, res in enumerate(priority_results):
            context_text += f"MATCH {i+1}: {res.payload.get('text')}\n\n"
            
        context_text += "COMPLETE DATABASE LIST:\n"
        for i, res in enumerate(all_results):
            p = res.payload
            context_text += (f"ID {i+1}: {p.get('team1')} vs {p.get('team2')} on {p.get('date')}. "
                             f"Winner: {p.get('winner')}. MoM: {p.get('mom')}. "
                             f"Top Scorer: {p.get('top_scorer')} ({p.get('top_scorer_runs')} runs).\n")
    else:
        search_result = client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=5,
        ).points
        context_text = ""
        for i, res in enumerate(search_result):
            context_text += f"MATCH RECORD #{i+1}:\n{res.payload.get('text', '')}\n\n"

    if not context_text:
        print("No relevant match data found.")
        continue

    prompt = f"""You are a strict Cricket Statistics Assistant. 
    INSTRUCTIONS:
    1. Answer ONLY using the DATABASE RECORDS provided below.
    2. If a player or result is NOT in the records, say "I don't have that information."
    3. Do NOT use outside knowledge.
    4. Base totals and win counts strictly on the IDs listed.

    RECORDS:
    {context_text.strip()}

    Question: {query}
    Answer:"""

    print("Querying Llama 3...")

    response = ollama.generate(
        model="llama3",
        prompt=prompt,
        options={
            "temperature": 0.0,
            "stop": ["MATCH RECORD #", "PRIORITY MATCH RECORDS"], 
            "num_predict": 700, 
        },
    )
    answer = response["response"].strip()

    print("\n" + "="*40)
    print("ANSWER:")
    print(answer)
    print("="*40)

client.close()
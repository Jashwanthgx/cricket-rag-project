from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, re, json, uuid
from dotenv import load_dotenv
from collections import Counter
from groq import Groq
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Access variables from .env
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION     = "cricket_pro"

groq_client = Groq(api_key=GROQ_API_KEY)

RAG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "rag-engine",
    "cricket_rag_data.jsonl"
)

qdrant_client: QdrantClient = None
embed_model: SentenceTransformer = None
total_docs: int = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant_client, embed_model, total_docs
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embed_model   = SentenceTransformer("all-MiniLM-L6-v2")
    try:
        info = qdrant_client.get_collection(COLLECTION)
        total_docs = info.points_count
    except Exception:
        total_docs = 0
    print(f"[startup] Connected to Qdrant. Collection has {total_docs} docs.")
    yield
    if qdrant_client:
        qdrant_client.close()

app = FastAPI(
    title="Cricket RAG API",
    description="Ask questions about cricket match data using RAG + Llama 3",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    query_type: str
    records_scanned: int

class IngestResponse(BaseModel):
    message: str
    total_ingested: int

AGGREGATE_PATTERNS = [
    r"\bhow many\b", r"\btotal\b", r"\bcount\b", r"\blist all\b",
    r"\bmost\b", r"\bhighest\b", r"\bbest\b", r"\baverage\b",
    r"\bevery\b", r"\bwho won\b", r"\bsummarize\b",
    r"\bcompare\b", r"\bdifference\b", r"\bboth\b", r"\bvs\b", r"\bversus\b"
]

def is_aggregate_query(q: str) -> bool:
    q_lower = q.lower()
    if any(word in q_lower for word in ["difference", "compare", "versus", "vs"]):
        return True
    if re.search(r"\bbetween\b", q_lower) and not any(p in q_lower for p in ["runs", "wickets", "stat"]):
        return False
    if any(re.search(p, q_lower) for p in AGGREGATE_PATTERNS):
        return True
    cricket_terms = ["runs", "wickets", "match", "played", "score", "century", "mom"]
    words = q.split()
    has_name   = any(w[0].isupper() for w in words[1:] if w)
    has_intent = any(t in q_lower for t in cricket_terms)
    return has_name and has_intent

def build_context(query_vector, query_type: str) -> tuple[str, int]:
    if query_type == "aggregate":
        priority = qdrant_client.query_points(
            collection_name=COLLECTION, query=query_vector, limit=5
        ).points

        all_res = qdrant_client.scroll(
            collection_name=COLLECTION,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )[0]

        win_counts           = Counter()
        mom_counts           = Counter()
        total_player_wickets = Counter()
        scorer_runs          = {}
        bowler_wickets       = {}
        bowler_economies     = {}
        chasing_wins         = Counter()

        for r in all_res:
            p       = r.payload
            winner  = p.get("winner", "")
            mom     = p.get("mom", "")
            scorer  = p.get("top_scorer", "")
            runs    = p.get("top_scorer_runs", 0)
            bowler  = p.get("top_bowler", "")
            wickets = p.get("top_bowler_wickets", 0)
            chased  = p.get("chased_win", False)

            if winner and winner not in ["No Result (Match Abandoned)", "Tie"]:
                win_counts[winner] += 1
            if mom:
                mom_counts[mom] += 1
            if scorer:
                if scorer not in scorer_runs or runs > scorer_runs[scorer]:
                    scorer_runs[scorer] = runs

            for b_name, stats in p.get("all_match_bowlers", {}).items():
                total_player_wickets[b_name] += stats.get("w", 0)
                econ = stats.get("e", 0.0)
                if econ is not None:
                    if b_name not in bowler_economies or econ < bowler_economies[b_name]:
                        bowler_economies[b_name] = econ

            if bowler:
                if bowler not in bowler_wickets or wickets > bowler_wickets[bowler]:
                    bowler_wickets[bowler] = wickets

            if winner and chased:
                chasing_wins[winner] += 1

        ctx = "PRE-COMPUTED STATISTICS (Use these for totals/most/comparisons):\n"
        ctx += "\nWIN COUNTS:\n"
        for team, count in win_counts.most_common():
            ctx += f"  {team}: {count} wins\n"

        ctx += "\nMAN OF THE MATCH COUNTS:\n"
        for player, count in mom_counts.most_common(50):
            ctx += f"  {player}: {count} awards\n"

        ctx += "\nTOTAL WICKETS ACROSS ALL MATCHES:\n"
        for player, count in total_player_wickets.most_common(50):
            ctx += f"  {player}: {count} wickets total\n"

        ctx += "\nBEST (LOWEST) SINGLE-MATCH ECONOMY:\n"
        for player, e in sorted(bowler_economies.items(), key=lambda x: x[1])[:50]:
            ctx += f"  {player}: {e} economy\n"

        ctx += "\nHIGHEST RUNS BY TOP SCORER:\n"
        for player, runs in sorted(scorer_runs.items(), key=lambda x: -x[1])[:50]:
            ctx += f"  {player}: {runs} runs\n"

        ctx += "\nBEST SINGLE-MATCH WICKET HAULS:\n"
        for player, wkts in sorted(bowler_wickets.items(), key=lambda x: -x[1])[:50]:
            ctx += f"  {player}: {wkts} wickets\n"

        ctx += "\nTEAMS WITH MOST WINS BATTING SECOND (CHASING):\n"
        for team, count in chasing_wins.most_common():
            ctx += f"  {team}: {count} wins while chasing\n"

        ctx += "\n─────────────────────────────────────\n"
        ctx += "\nPRIORITY MATCH RECORDS (FULL DETAIL):\n"
        for i, r in enumerate(priority):
            ctx += f"MATCH {i+1}:\n{r.payload.get('text')}\n\n"

        return ctx, len(all_res)

    else:
        results = qdrant_client.query_points(
            collection_name=COLLECTION, query=query_vector, limit=5
        ).points
        ctx = ""
        for i, r in enumerate(results):
            ctx += f"MATCH RECORD #{i+1}:\n{r.payload.get('text', '')}\n\n"
        return ctx, len(results)

def ask_llama(context: str, question: str, query_type: str) -> str:
    prompt = f"""You are a Cricket Statistics Assistant.

CRITICAL RULES:
- You MUST answer using ONLY the PRE-COMPUTED STATISTICS section below.
- For ANY question about "most", "highest", "best", "total" — look at PRE-COMPUTED STATISTICS FIRST.
- NEVER say "I don't have that information" if the answer exists in PRE-COMPUTED STATISTICS.
- Give the answer directly. Do not explain your reasoning.

{context.strip()}

Question: {question}
Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a cricket statistics assistant.
IMPORTANT: The context always contains a PRE-COMPUTED STATISTICS section.
For questions about totals, most, highest, best — READ THAT SECTION and answer directly.
Never say you don't have information if it appears in PRE-COMPUTED STATISTICS."""
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()

@app.get("/")
def root():
    return {"status": "ok", "message": "Cricket RAG API is running 🏏"}

@app.get("/health")
def health():
    try:
        info = qdrant_client.get_collection(COLLECTION)
        return {"status": "healthy", "collection": COLLECTION, "docs": info.points_count}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    query_type   = "aggregate" if is_aggregate_query(req.question) else "semantic"
    query_vector = embed_model.encode(req.question).tolist()
    context, scanned = build_context(query_vector, query_type)

    if not context.strip():
        raise HTTPException(status_code=404, detail="No relevant match data found.")

    answer = await run_in_threadpool(ask_llama, context, req.question, query_type)

    return QueryResponse(
        question=req.question,
        answer=answer,
        query_type=query_type,
        records_scanned=scanned
    )

@app.post("/ingest", response_model=IngestResponse)
def ingest():
    if not os.path.exists(RAG_FILE):
        raise HTTPException(status_code=404, detail=f"'{RAG_FILE}' not found.")

    documents, payloads = [], []
    with open(RAG_FILE) as f:
        for line in f:
            d = json.loads(line)
            documents.append(d["text"])
            payloads.append({
                "text":               d["text"],
                "team1":              d.get("team1", ""),
                "team2":              d.get("team2", ""),
                "date":               d.get("date", ""),
                "venue":              d.get("venue", ""),
                "winner":             d.get("winner", ""),
                "mom":                d.get("mom", ""),
                "top_scorer":         d.get("top_scorer", "N/A"),
                "top_scorer_runs":    d.get("top_scorer_runs", 0),
                "top_bowler":         d.get("top_bowler", "N/A"),
                "top_bowler_wickets": d.get("top_bowler_wickets", 0),
                "top_bowler_economy": d.get("top_bowler_economy", 0.0),
                "all_match_bowlers":  d.get("all_match_bowlers", {}),
                "chased_win":         d.get("chased_win", False),
            })

    embeddings = embed_model.encode(documents, show_progress_bar=False)

    try:
        qdrant_client.delete_collection(COLLECTION)
    except Exception:
        pass

    qdrant_client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
    )

    points = [
        PointStruct(id=str(uuid.uuid4()), vector=embeddings[i].tolist(), payload=payloads[i])
        for i in range(len(documents))
    ]

    for start in range(0, len(points), 50):
        qdrant_client.upsert(collection_name=COLLECTION, points=points[start:start+50])

    global total_docs
    total_docs = len(documents)

    return IngestResponse(message="Ingestion complete.", total_ingested=len(documents))
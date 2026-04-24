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
        total_player_runs    = Counter()
        total_player_balls   = Counter()
        total_player_fours   = Counter()
        total_player_sixes   = Counter()
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

            # Aggregate from all_match_batters list
            for batter in p.get("all_match_batters", []):
                name = batter.get("name", "")
                if name:
                    total_player_runs[name] += batter.get("runs", 0)
                    total_player_balls[name] += batter.get("balls", 0)
                    total_player_fours[name] += batter.get("fours", 0)
                    total_player_sixes[name] += batter.get("sixes", 0)

            # Aggregate from all_match_bowlers list
            for bowler_stat in p.get("all_match_bowlers", []):
                name = bowler_stat.get("name", "")
                if name:
                    total_player_wickets[name] += bowler_stat.get("wickets", 0)
                    econ = bowler_stat.get("economy", 0.0)
                    if econ is not None and econ > 0:
                        if name not in bowler_economies or econ < bowler_economies[name]:
                            bowler_economies[name] = econ

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

        ctx += "\nTOTAL RUNS ACROSS ALL MATCHES:\n"
        for player, runs in total_player_runs.most_common(50):
            balls = total_player_balls.get(player, 0)
            sr = round((runs / balls * 100), 2) if balls > 0 else 0
            fours = total_player_fours.get(player, 0)
            sixes = total_player_sixes.get(player, 0)
            ctx += f"  {player}: {runs} runs ({balls} balls, SR: {sr}, 4s: {fours}, 6s: {sixes})\n"

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
                "all_match_batters":  d.get("all_match_batters", []),
                "all_match_bowlers":  d.get("all_match_bowlers", []),
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


# ==================== ANALYTICS ENDPOINTS ====================

class PlayerComparisonRequest(BaseModel):
    player1: str
    player2: str

class PlayerComparisonResponse(BaseModel):
    player1: str
    player2: str
    player1_stats: dict
    player2_stats: dict
    comparison: str

class CareerTotalsRequest(BaseModel):
    player_name: str

class CareerTotalsResponse(BaseModel):
    player_name: str
    batting: dict
    bowling: dict
    matches_played: int

class HeadToHeadRequest(BaseModel):
    batter_name: str
    bowler_name: str

class HeadToHeadResponse(BaseModel):
    batter_name: str
    bowler_name: str
    dismissals: int
    runs_scored: int
    balls_faced: int
    times_out: int
    matches: int


def aggregate_player_stats(all_matches: list) -> dict:
    """Aggregate career stats for all players from match data."""
    player_career = {}

    for r in all_matches:
        p = r.payload

        # Aggregate batting stats
        for batter in p.get("all_match_batters", []):
            name = batter.get("name", "")
            if name not in player_career:
                player_career[name] = {
                    "batting": {"runs": 0, "balls": 0, "fours": 0, "sixes": 0, "innings": 0},
                    "bowling": {"wickets": 0, "runs_conceded": 0, "balls_bowled": 0, "innings": 0},
                    "matches": set()
                }
            player_career[name]["batting"]["runs"] += batter.get("runs", 0)
            player_career[name]["batting"]["balls"] += batter.get("balls", 0)
            player_career[name]["batting"]["fours"] += batter.get("fours", 0)
            player_career[name]["batting"]["sixes"] += batter.get("sixes", 0)
            player_career[name]["batting"]["innings"] += 1
            player_career[name]["matches"].add(p.get("date", "") + p.get("team1", "") + p.get("team2", ""))

        # Aggregate bowling stats
        for bowler in p.get("all_match_bowlers", []):
            name = bowler.get("name", "")
            if name not in player_career:
                player_career[name] = {
                    "batting": {"runs": 0, "balls": 0, "fours": 0, "sixes": 0, "innings": 0},
                    "bowling": {"wickets": 0, "runs_conceded": 0, "balls_bowled": 0, "innings": 0},
                    "matches": set()
                }
            player_career[name]["bowling"]["wickets"] += bowler.get("wickets", 0)
            player_career[name]["bowling"]["runs_conceded"] += bowler.get("runs_conceded", 0)
            player_career[name]["bowling"]["balls_bowled"] += bowler.get("balls_bowled", 0)
            player_career[name]["bowling"]["innings"] += 1
            player_career[name]["matches"].add(p.get("date", "") + p.get("team1", "") + p.get("team2", ""))

    # Convert sets to counts and calculate derived stats
    for name, stats in player_career.items():
        stats["matches_played"] = len(stats["matches"])
        del stats["matches"]

        # Batting derived stats
        balls = stats["batting"]["balls"]
        runs = stats["batting"]["runs"]
        stats["batting"]["strike_rate"] = round((runs / balls * 100), 2) if balls > 0 else 0.0
        stats["batting"]["average"] = round(runs / max(1, stats["batting"]["innings"]), 2)

        # Bowling derived stats
        overs = stats["bowling"]["balls_bowled"] / 6
        wickets = stats["bowling"]["wickets"]
        runs_conceded = stats["bowling"]["runs_conceded"]
        stats["bowling"]["economy"] = round(runs_conceded / overs, 2) if overs > 0 else 0.0
        stats["bowling"]["average"] = round(runs_conceded / max(1, wickets), 2)

    return player_career


@app.post("/analytics/compare-players", response_model=PlayerComparisonResponse)
async def compare_players(req: PlayerComparisonRequest):
    """Compare career statistics between two players."""
    all_matches = qdrant_client.scroll(
        collection_name=COLLECTION, limit=10000, with_payload=True, with_vectors=False
    )[0]

    player_stats = aggregate_player_stats(all_matches)

    p1_stats = player_stats.get(req.player1, {
        "batting": {"runs": 0, "balls": 0, "fours": 0, "sixes": 0, "innings": 0, "strike_rate": 0, "average": 0},
        "bowling": {"wickets": 0, "runs_conceded": 0, "balls_bowled": 0, "innings": 0, "economy": 0, "average": 0},
        "matches_played": 0
    })

    p2_stats = player_stats.get(req.player2, {
        "batting": {"runs": 0, "balls": 0, "fours": 0, "sixes": 0, "innings": 0, "strike_rate": 0, "average": 0},
        "bowling": {"wickets": 0, "runs_conceded": 0, "balls_bowled": 0, "innings": 0, "economy": 0, "average": 0},
        "matches_played": 0
    })

    comparison = []
    if p1_stats["batting"]["runs"] > p2_stats["batting"]["runs"]:
        comparison.append(f"{req.player1} has scored more career runs ({p1_stats['batting']['runs']} vs {p2_stats['batting']['runs']})")
    elif p2_stats["batting"]["runs"] > p1_stats["batting"]["runs"]:
        comparison.append(f"{req.player2} has scored more career runs ({p2_stats['batting']['runs']} vs {p1_stats['batting']['runs']})")

    if p1_stats["bowling"]["wickets"] > p2_stats["bowling"]["wickets"]:
        comparison.append(f"{req.player1} has taken more wickets ({p1_stats['bowling']['wickets']} vs {p2_stats['bowling']['wickets']})")
    elif p2_stats["bowling"]["wickets"] > p1_stats["bowling"]["wickets"]:
        comparison.append(f"{req.player2} has taken more wickets ({p2_stats['bowling']['wickets']} vs {p1_stats['bowling']['wickets']})")

    if p1_stats["batting"]["strike_rate"] > p2_stats["batting"]["strike_rate"] and p1_stats["batting"]["balls"] > 0:
        comparison.append(f"{req.player1} has a better strike rate ({p1_stats['batting']['strike_rate']} vs {p2_stats['batting']['strike_rate']})")
    elif p2_stats["batting"]["strike_rate"] > p1_stats["batting"]["strike_rate"] and p2_stats["batting"]["balls"] > 0:
        comparison.append(f"{req.player2} has a better strike rate ({p2_stats['batting']['strike_rate']} vs {p1_stats['batting']['strike_rate']})")

    return PlayerComparisonResponse(
        player1=req.player1,
        player2=req.player2,
        player1_stats=p1_stats,
        player2_stats=p2_stats,
        comparison=" | ".join(comparison) if comparison else "Players have similar statistics"
    )


@app.post("/analytics/career-totals", response_model=CareerTotalsResponse)
async def get_career_totals(req: CareerTotalsRequest):
    """Get lifetime career totals for a specific player."""
    all_matches = qdrant_client.scroll(
        collection_name=COLLECTION, limit=10000, with_payload=True, with_vectors=False
    )[0]

    player_stats = aggregate_player_stats(all_matches)

    stats = player_stats.get(req.player_name, {
        "batting": {"runs": 0, "balls": 0, "fours": 0, "sixes": 0, "innings": 0, "strike_rate": 0, "average": 0},
        "bowling": {"wickets": 0, "runs_conceded": 0, "balls_bowled": 0, "innings": 0, "economy": 0, "average": 0},
        "matches_played": 0
    })

    return CareerTotalsResponse(
        player_name=req.player_name,
        batting=stats["batting"],
        bowling=stats["bowling"],
        matches_played=stats["matches_played"]
    )


@app.post("/analytics/head-to-head", response_model=HeadToHeadResponse)
async def get_batter_vs_bowler(req: HeadToHeadRequest):
    """Get head-to-head stats between a batter and bowler."""
    all_matches = qdrant_client.scroll(
        collection_name=COLLECTION, limit=10000, with_payload=True, with_vectors=False
    )[0]

    total_runs = 0
    total_balls = 0
    total_dismissals = 0
    matches_faced = 0

    for r in all_matches:
        p = r.payload
        match_found = False

        # Check if batter faced this bowler in this match
        for batter in p.get("all_match_batters", []):
            if batter.get("name", "").lower() == req.batter_name.lower():
                match_found = True
                total_runs += batter.get("runs", 0)
                total_balls += batter.get("balls", 0)

        if match_found:
            matches_faced += 1

        # Count dismissals (simplified - counts bowler wickets when both players in same match)
        for bowler in p.get("all_match_bowlers", []):
            if bowler.get("name", "").lower() == req.bowler_name.lower() and match_found:
                total_dismissals += bowler.get("wickets", 0)

    # Cap dismissals at balls faced (sanity check)
    times_out = min(total_dismissals, total_balls)

    return HeadToHeadResponse(
        batter_name=req.batter_name,
        bowler_name=req.bowler_name,
        dismissals=times_out,
        runs_scored=total_runs,
        balls_faced=total_balls,
        times_out=times_out,
        matches=matches_faced
    )


@app.get("/analytics/leaderboard/batting")
async def get_batting_leaderboard(limit: int = 20):
    """Get top batters by runs."""
    all_matches = qdrant_client.scroll(
        collection_name=COLLECTION, limit=10000, with_payload=True, with_vectors=False
    )[0]

    player_stats = aggregate_player_stats(all_matches)

    leaderboard = []
    for name, stats in player_stats.items():
        if stats["batting"]["runs"] > 0:
            leaderboard.append({
                "name": name,
                "runs": stats["batting"]["runs"],
                "innings": stats["batting"]["innings"],
                "strike_rate": stats["batting"]["strike_rate"],
                "average": stats["batting"]["average"],
                "fours": stats["batting"]["fours"],
                "sixes": stats["batting"]["sixes"]
            })

    leaderboard.sort(key=lambda x: x["runs"], reverse=True)
    return {"leaderboard": leaderboard[:limit]}


@app.get("/analytics/leaderboard/bowling")
async def get_bowling_leaderboard(limit: int = 20):
    """Get top bowlers by wickets."""
    all_matches = qdrant_client.scroll(
        collection_name=COLLECTION, limit=10000, with_payload=True, with_vectors=False
    )[0]

    player_stats = aggregate_player_stats(all_matches)

    leaderboard = []
    for name, stats in player_stats.items():
        if stats["bowling"]["wickets"] > 0:
            leaderboard.append({
                "name": name,
                "wickets": stats["bowling"]["wickets"],
                "innings": stats["bowling"]["innings"],
                "economy": stats["bowling"]["economy"],
                "average": stats["bowling"]["average"],
                "runs_conceded": stats["bowling"]["runs_conceded"]
            })

    leaderboard.sort(key=lambda x: x["wickets"], reverse=True)
    return {"leaderboard": leaderboard[:limit]}
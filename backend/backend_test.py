from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, re, json, uuid
import sys
from datetime import datetime
from dotenv import load_dotenv
from collections import Counter, defaultdict
from groq import Groq
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue, Range, PayloadSchemaType
)
from sentence_transformers import SentenceTransformer
from typing import Optional
import torch
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from rag_engine.utils import (
    GROQ_MODEL,
    PHASE_KEYWORDS, STYLE_KEYWORDS, AGGREGATE_PATTERNS, NAME_ALIASES,
    detect_phase, detect_style, normalize_query,
    is_aggregate_query, build_qdrant_filter,
    aggregate_pace_vs_spin_by_phase, pace_vs_spin_context_str,
    filter_points_by_phase,
    aggregate_mom_awards, mom_context_str,
)

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"[gpu] CUDA available: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory // 1024**2} MB VRAM)")
else:
    print("[gpu] No CUDA GPU detected — running on CPU.")

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION     = "cricket_pro_v2"

groq_client = Groq(api_key=GROQ_API_KEY)

RAG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cricket_rag_data.jsonl")

qdrant_client: QdrantClient = None
embed_model:   SentenceTransformer = None
total_docs:    int = 0


_stats_cache: dict = {}                   
_stats_cache_built_at: Optional[datetime] = None
_bowler_points_cache: list = []           
_cache_lock = __import__("threading").Lock()  


def _build_stats_cache() -> None:
    """Scroll the full collection once and populate _stats_cache."""
    global _stats_cache, _stats_cache_built_at, _bowler_points_cache
    if qdrant_client is None:
        return
    import time
    MAX_CACHE_RETRIES = 3
    for cache_attempt in range(MAX_CACHE_RETRIES):
        try:
            with _cache_lock:
                print(f"[cache] Build attempt {cache_attempt + 1}/{MAX_CACHE_RETRIES}...")
                all_points = _scroll_all(role_filter=None)
                _stats_cache = aggregate_player_stats(all_points)
                _stats_cache_built_at = datetime.utcnow()
                print(f"[cache] Stats cache built for {len(_stats_cache)} players.")
                _bowler_points_cache = _scroll_all(role_filter="bowler")
                print(f"[cache] Bowler points cache built: {len(_bowler_points_cache)} records.")
            break
        except Exception as e:
            wait = 10 * (2 ** cache_attempt)
            print(f"[cache] Failed to build stats cache (attempt {cache_attempt + 1}): {e}")
            if cache_attempt < MAX_CACHE_RETRIES - 1:
                print(f"[cache] Retrying in {wait}s...")
                time.sleep(wait)


def _scroll_all(role_filter: Optional[str] = None, player_filter: Optional[str] = None) -> list:
    """Paginated scroll through the collection with optional role or player filters."""
    import time
    MAX_RETRIES = 5
    BASE_DELAY  = 2

    conditions = []
    if role_filter:
        conditions.append(FieldCondition(key="role", match=MatchValue(value=role_filter)))
    if player_filter:
        conditions.append(FieldCondition(key="player_name", match=MatchValue(value=player_filter)))
    
    scroll_filter = Filter(must=conditions) if conditions else None

    results = []
    offset  = None
    while True:
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                batch, next_offset = qdrant_client.scroll(
                    collection_name=COLLECTION,
                    limit=5000,  
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                    scroll_filter=scroll_filter,
                )
                break
            except Exception as exc:
                last_exc = exc
                wait = BASE_DELAY * (2 ** attempt)
                time.sleep(wait)
        else:
            raise RuntimeError(f"Scroll failed after {MAX_RETRIES} attempts.")

        results.extend(batch)
        if next_offset is None:
            break
        offset = next_offset
    return results


def _ensure_payload_indexes() -> None:
    """Create all required payload indexes, including the new player_name index[cite: 6]."""
    keyword_fields = ["role", "metadata.bowler_style", "format", "player_name"]
    integer_fields = [
        "phase_stats.powerplay.balls",
        "phase_stats.middle.balls",
        "phase_stats.death.balls",
    ]
    for field in keyword_fields:
        try:
            qdrant_client.create_payload_index(COLLECTION, field, PayloadSchemaType.KEYWORD)
        except Exception: pass
    for field in integer_fields:
        try:
            qdrant_client.create_payload_index(COLLECTION, field, PayloadSchemaType.INTEGER)
        except Exception: pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant_client, embed_model, total_docs
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=300)
    embed_model   = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    try:
        info = qdrant_client.get_collection(COLLECTION)
        total_docs = info.points_count
    except Exception: total_docs = 0
    _ensure_payload_indexes()
    import threading
    threading.Thread(target=_build_stats_cache, daemon=True).start()
    yield
    if qdrant_client: qdrant_client.close()

app = FastAPI(title="Cricket RAG API v3", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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

class PaceVsSpinRequest(BaseModel):
    phase: str = "death"

class ComparePlayersRequest(BaseModel):
    player1: str
    player2: str

class CareerTotalsRequest(BaseModel):
    player_name: str


def detect_margin_query(question: str) -> Optional[dict]:
    """
    Detect if the query is asking for matches with a specific margin.
    Returns dict with 'method' and 'margin' if found, else None.
    """
    q = question.lower()

   
    import re
    margin_pattern = r"won by (\d+) (runs|wickets|innings)"
    match = re.search(margin_pattern, q)

    if match:
        return {
            "margin": int(match.group(1)),
            "method": match.group(2)
        }
    return None

def detect_venue_query(question: str) -> Optional[str]:
    """
    Detect if the query is asking about a specific venue.
    Returns venue name if found, else None.
    """
    q = question.lower()

    
    venues = {
        "wankhede": ["wankhede", "mumbai"],
        "lord's": ["lord's", "lords"],
        "melbourne cricket ground": ["melbourne cricket ground", "mcg", "melbourne"],
        "eden gardens": ["eden gardens", "kolkata"],
        "chinnaswamy": ["chinnaswamy", "bangalore"],
        "chepauk": ["chepauk", "chennai"],
        "sydney cricket ground": ["sydney cricket ground", "scg", "sydney"],
        "perth": ["perth", "waca"],
        "bristol": ["bristol"],
        "the oval": ["the oval", "kennington", "oval"],
        "headingley": ["headingley", "leeds"],
        "trent bridge": ["trent bridge", "nottingham"],
        "old trafford": ["old trafford", "manchester"],
        "harare": ["harare"],
        "dubai": ["dubai"],
        "abu dhabi": ["abu dhabi"],
        "sharjah": ["sharjah"],
        "colombo": ["colombo"],
        "kandy": ["kandy"],
        "galle": ["galle"],
        "moratuwa": ["moratuwa"],
        "dhaka": ["dhaka"],
        "mirpur": ["mirpur"],
        "chittagong": ["chittagong"],
        "karachi": ["karachi"],
        "lahore": ["lahore"],
        "multan": ["multan"],
        "rawalpindi": ["rawalpindi"],
        "faisalabad": ["faisalabad"],
        "peshawar": ["peshawar"],
    }

    for venue, variations in venues.items():
        for variation in variations:
            if variation in q:
                return venue
    return None


def build_context(
    query_vector: list,
    query_type: str,
    question: str,
    detected_player: Optional[str] = None 
) -> tuple[str, int]:
    detected_phase = detect_phase(question)
    detected_style = detect_style(question)
    detected_margin = detect_margin_query(question)
    detected_venue = detect_venue_query(question)
    query_filter   = build_qdrant_filter(detected_phase, detected_style, detected_player)

    if query_type == "aggregate":
        
        all_res = _scroll_all(player_filter=detected_player) if detected_player else _scroll_all()
        
        
        if detected_style:
             all_res = [r for r in all_res if r.payload.get("metadata", {}).get("bowler_style") == detected_style]

        total_player_runs     = Counter()
        total_player_balls    = Counter()
        total_player_wickets  = Counter()
        runs_conceded         = Counter()
        balls_bowled          = Counter()
        batting_inns          = Counter()
        bowling_inns          = Counter()

        
        match_outcomes = []  
        team_wins_by_method = defaultdict(lambda: defaultdict(int))  

        for r in all_res:
            p, name, role, meta = r.payload, r.payload.get("player_name", ""), r.payload.get("role", ""), r.payload.get("metadata", {})

            if role == "match":
                
                winner = meta.get("match_winner", "")
                win_margin = meta.get("win_margin", 0)
                win_method = meta.get("win_method", "")
                team = meta.get("team", "")
                opponent = meta.get("opponent", "")
                venue = meta.get("venue", "")
                date = meta.get("date", "")
                mom = meta.get("man_of_match", "")

                if winner:
                    match_outcomes.append({
                        "teams": f"{team} vs {opponent}",
                        "winner": winner,
                        "margin": win_margin,
                        "method": win_method,
                        "venue": venue,
                        "date": date,
                        "mom": mom,
                    })
                    team_wins_by_method[win_method][winner] += 1
            elif role == "batter":
                total_player_runs[name] += meta.get("total_runs", 0)
                total_player_balls[name] += meta.get("total_balls", 0)
                batting_inns[name] += 1
            elif role == "bowler":
                total_player_wickets[name] += meta.get("total_wickets", 0)
                runs_conceded[name] += meta.get("total_runs", 0)
                balls_bowled[name] += meta.get("total_balls", 0)
                bowling_inns[name] += 1

        
        team_wins = Counter()
        for method, teams in team_wins_by_method.items():
            for team, wins in teams.items():
                team_wins[team] += wins

        ctx = "PRE-COMPUTED STATISTICS:\n\nTOTAL RUNS:\n"
        for player, runs in total_player_runs.most_common(50):
            ctx += f"  {player}: {runs} runs ({batting_inns[player]} inns)\n"

        ctx += "\nTOTAL WICKETS:\n"
        for player, wkts in total_player_wickets.most_common(50):
            ctx += f"  {player}: {wkts} wkts\n"

       
        ctx += mom_context_str(all_res, top_n=20)

        
        if team_wins_by_method:
            ctx += "\nTEAM WINS BY METHOD:\n"
            for method in ["runs", "wickets", "DLS", "innings"]:
                if method in team_wins_by_method:
                    total_wins = sum(team_wins_by_method[method].values())
                    ctx += f"  {method.upper()}: {total_wins} total wins\n"
                    for team, wins in sorted(team_wins_by_method[method].items(), key=lambda x: -x[1])[:10]:
                        ctx += f"    {team}: {wins} wins\n"

        
        if team_wins:
            ctx += "\nTOTAL TEAM WINS:\n"
            for team, wins in team_wins.most_common(20):
                ctx += f"  {team}: {wins} wins\n"

        
        if match_outcomes:
            ctx += "\nDETAILED MATCH OUTCOMES:\n"
            
            matches_by_method = defaultdict(list)
            for match in match_outcomes[:50]:  
                matches_by_method[match['method']].append(match)

            for method in ["runs", "wickets", "DLS", "innings"]:
                if method in matches_by_method:
                    ctx += f"  {method.upper()}:\n"
                    for match in matches_by_method[method][:15]:
                        ctx += f"    {match['teams']}: {match['winner']} won by {match['margin']} {method} ({match['date']})\n"

        
        if detected_margin and match_outcomes:
            ctx += f"\nMATCHES WON BY EXACTLY {detected_margin['margin']} {detected_margin['method'].upper()}:\n"
            filtered_matches = [
                m for m in match_outcomes
                if m['method'] == detected_margin['method'] and m['margin'] == detected_margin['margin']
            ]
            if filtered_matches:
                for match in filtered_matches[:20]:
                    ctx += f"  {match['teams']}: {match['winner']} won by {match['margin']} {match['method']} ({match['date']}, {match['venue']})\n"
            else:
                ctx += f"  No matches found with exactly {detected_margin['margin']} {detected_margin['method']} margin.\n"

        
        if detected_venue and match_outcomes:
            ctx += f"\nMATCHES AT {detected_venue.upper()}:\n"
            filtered_matches = [
                m for m in match_outcomes
                if detected_venue.lower() in m['venue'].lower()
            ]
            if filtered_matches:
                for match in filtered_matches[:15]:
                    ctx += f"  {match['teams']}: {match['winner']} won by {match['margin']} {match['method']} ({match['date']})\n"
            else:
                ctx += f"  No matches found at {detected_venue}.\n"

        
        if detected_venue and detected_style:
            ctx += f"\nBOWLER STATISTICS AT {detected_venue.upper()} ({detected_style.upper()}):\n"
            venue_bowlers = []
            for r in all_res:
                p, name, role, meta = r.payload, r.payload.get("player_name", ""), r.payload.get("role", ""), r.payload.get("metadata", {})
                if role == "bowler" and detected_venue.lower() in meta.get("venue", "").lower():
                    style = meta.get("bowler_style", "")
                    if style.lower() == detected_style.lower():
                        venue_bowlers.append({
                            "name": name,
                            "wickets": meta.get("total_wickets", 0),
                            "runs": meta.get("total_runs", 0),
                            "balls": meta.get("total_balls", 0),
                            "economy": meta.get("economy", 0.0),
                            "phase": detected_phase if detected_phase else "all"
                        })

            if venue_bowlers:
                
                venue_bowlers.sort(key=lambda x: x["economy"])
                for bowler in venue_bowlers[:10]:
                    overs = round(bowler["balls"] / 6, 1) if bowler["balls"] > 0 else 0.0
                    ctx += f"  {bowler['name']}: {bowler['wickets']} wkts, {overs} ov, Econ {bowler['economy']}\n"
            else:
                ctx += f"  No {detected_style} bowlers found at {detected_venue}.\n"

        if detected_phase or detected_style:
            ctx += pace_vs_spin_context_str(all_res, detected_phase)

        ctx += "\nPRIORITY RECORDS:\n"
        priority = qdrant_client.query_points(COLLECTION, query=query_vector, limit=5, query_filter=query_filter).points
        for i, r in enumerate(priority): ctx += f"RECORD #{i + 1}: {r.payload.get('text')}\n\n"
        return ctx, len(all_res)

    else:
        results = qdrant_client.query_points(COLLECTION, query=query_vector, limit=5, query_filter=query_filter).points
        ctx = ""
        for i, r in enumerate(results): ctx += f"RECORD #{i + 1}:\n{r.payload.get('text', '')}\n\n"
        return ctx, len(results)


def ask_llama(context: str, question: str, query_type: str) -> str:
    prompt = f"""You are a cricket statistics expert. Answer the question using ONLY the statistics provided below.

Available data includes:
- Player statistics (runs, wickets, batting/bowling averages, strike rates, economy)
- Man of the Match awards with complete history
- Team wins by method (runs, wickets, DLS, innings) with totals and team breakdowns
- Detailed match outcomes with winners, margins, dates, and venues
- Phase-specific statistics (powerplay, middle, death)
- Venue-specific match outcomes and bowler statistics

For questions about specific margins (e.g., "won by exactly 1 wicket"), check the "MATCHES WON BY EXACTLY X Y" section.

For venue-specific questions, check the "MATCHES AT VENUE" and "BOWLER STATISTICS AT VENUE" sections.

For team performance questions, check "TEAM WINS BY METHOD" and "TOTAL TEAM WINS" sections.

For player awards, check "MAN OF THE MATCH AWARDS" section.

If the information is not available in the statistics, say "I don't have that information in the dataset."

STATISTICS:
{context.strip()}

Question: {question}
Answer:"""
    response = groq_client.chat.completions.create(model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.0)
    return response.choices[0].message.content.strip()


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.question.strip(): raise HTTPException(status_code=400)
    question = normalize_query(req.question)
    detected_player = next((val for val in NAME_ALIASES.values() if val in question), None)
    query_type = "aggregate" if is_aggregate_query(question) else "semantic"
    query_vector = embed_model.encode(question, convert_to_numpy=True, normalize_embeddings=True).tolist()
    
    
    context, scanned = await run_in_threadpool(build_context, query_vector, query_type, question, detected_player)
    answer = await run_in_threadpool(ask_llama, context, req.question, query_type)
    return QueryResponse(question=question, answer=answer, query_type=query_type, records_scanned=scanned)

@app.post("/ingest", response_model=IngestResponse)
def ingest():
    if not os.path.exists(RAG_FILE): raise HTTPException(status_code=404)
    documents, payloads = [], []
    with open(RAG_FILE) as f:
        for line in f:
            d = json.loads(line)
            documents.append(d["text"]); payloads.append(d)

    
    ENCODE_BATCH = 2048 if DEVICE == "cuda" else 512
    all_embeddings = []
    for start in range(0, len(documents), ENCODE_BATCH):
        batch = documents[start:start + ENCODE_BATCH]
        all_embeddings.append(embed_model.encode(batch, convert_to_numpy=True, normalize_embeddings=True))
    embeddings = np.concatenate(all_embeddings, axis=0)

    try: qdrant_client.delete_collection(COLLECTION)
    except Exception: pass
    qdrant_client.create_collection(COLLECTION, vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE))
    _ensure_payload_indexes()

    points = [PointStruct(id=str(uuid.uuid4()), vector=embeddings[i].tolist(), payload=payloads[i]) for i in range(len(documents))]
    for start in range(0, len(points), 250):
        qdrant_client.upsert(COLLECTION, points=points[start:start + 250])
    
    _build_stats_cache()
    return IngestResponse(message="Ingestion complete.", total_ingested=len(documents))


def aggregate_player_stats(all_matches: list) -> dict:
    player_career = {}
    for r in all_matches:
        p, name, role, meta = r.payload, r.payload.get("player_name", ""), r.payload.get("role", ""), r.payload.get("metadata", {})
        if not name: continue
        
        if role == "match": continue
        if name not in player_career:
            player_career[name] = {
                "batting": {"runs": 0, "balls": 0, "innings": 0},
                "bowling": {"wickets": 0, "runs_conceded": 0, "balls_bowled": 0, "innings": 0},
                "matches_played": 0
            }
        player_career[name]["matches_played"] += 1
        if role == "batter":
            player_career[name]["batting"]["runs"] += meta.get("total_runs", 0)
            player_career[name]["batting"]["balls"] += meta.get("total_balls", 0)
            player_career[name]["batting"]["innings"] += 1
        elif role == "bowler":
            player_career[name]["bowling"]["wickets"] += meta.get("total_wickets", 0)
            player_career[name]["bowling"]["runs_conceded"] += meta.get("total_runs", 0)
            player_career[name]["bowling"]["balls_bowled"] += meta.get("total_balls", 0)
            player_career[name]["bowling"]["innings"] += 1
    return player_career

@app.post("/analytics/career-totals")
async def get_career_totals(req: CareerTotalsRequest):
    player_name = normalize_query(req.player_name)
    stats = _stats_cache.get(player_name, {})

    
    batting = stats.get("batting", {})
    bowling = stats.get("bowling", {})

    batting_avg = round(batting.get("runs", 0) / max(1, batting.get("innings", 1)), 2)
    batting_sr = round((batting.get("runs", 0) / max(1, batting.get("balls", 1))) * 100, 2) if batting.get("balls", 0) > 0 else 0.0

    bowling_avg = round(bowling.get("runs_conceded", 0) / max(1, bowling.get("wickets", 1)), 2)
    bowling_econ = round(bowling.get("runs_conceded", 0) / max(1, bowling.get("balls_bowled", 1) / 6), 2) if bowling.get("balls_bowled", 0) > 0 else 0.0

    return {
        "player_name": player_name,
        "matches_played": stats.get("matches_played", 0),
        "batting": {
            "runs": batting.get("runs", 0),
            "innings": batting.get("innings", 0),
            "average": batting_avg,
            "strike_rate": batting_sr,
        },
        "bowling": {
            "wickets": bowling.get("wickets", 0),
            "innings": bowling.get("innings", 0),
            "average": bowling_avg,
            "economy": bowling_econ,
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint for frontend status indicator."""
    try:
        info = qdrant_client.get_collection(COLLECTION)
        return {"status": "healthy", "docs": info.points_count}
    except Exception:
        return {"status": "unhealthy", "docs": 0}


@app.get("/analytics/leaderboard/batting")
async def get_batting_leaderboard(limit: int = 15):
    """Return top run scorers with detailed batting stats."""
    batting_stats = []
    for name, stats in _stats_cache.items():
        batting = stats.get("batting", {})
        if batting.get("innings", 0) > 0:
            runs = batting.get("runs", 0)
            balls = batting.get("balls", 0)
            innings = batting.get("innings", 0)

            
            average = round(runs / max(1, innings), 2)
            strike_rate = round((runs / max(1, balls)) * 100, 2) if balls > 0 else 0.0

            
            fours = 0
            sixes = 0

            batting_stats.append({
                "name": name,
                "runs": runs,
                "innings": innings,
                "average": average,
                "strike_rate": strike_rate,
                "fours": fours,
                "sixes": sixes,
            })

    
    batting_stats.sort(key=lambda x: x["runs"], reverse=True)
    return {"leaderboard": batting_stats[:limit]}


@app.get("/analytics/leaderboard/bowling")
async def get_bowling_leaderboard(limit: int = 15):
    """Return top wicket takers with detailed bowling stats."""
    bowling_stats = []
    for name, stats in _stats_cache.items():
        bowling = stats.get("bowling", {})
        if bowling.get("innings", 0) > 0:
            wickets = bowling.get("wickets", 0)
            runs_conceded = bowling.get("runs_conceded", 0)
            balls_bowled = bowling.get("balls_bowled", 0)
            innings = bowling.get("innings", 0)

            
            average = round(runs_conceded / max(1, wickets), 2) if wickets > 0 else 0.0
            economy = round(runs_conceded / max(1, balls_bowled / 6), 2) if balls_bowled > 0 else 0.0

            bowling_stats.append({
                "name": name,
                "wickets": wickets,
                "innings": innings,
                "average": average,
                "economy": economy,
                "runs_conceded": runs_conceded,
            })

    
    bowling_stats.sort(key=lambda x: x["wickets"], reverse=True)
    return {"leaderboard": bowling_stats[:limit]}


@app.post("/analytics/compare-players")
async def compare_players(req: ComparePlayersRequest):
    """Compare two players head-to-head with detailed stats."""
    player1 = normalize_query(req.player1)
    player2 = normalize_query(req.player2)

    stats1 = _stats_cache.get(player1, {})
    stats2 = _stats_cache.get(player2, {})

    
    bat1 = stats1.get("batting", {})
    bowl1 = stats1.get("bowling", {})
    player1_stats = {
        "batting": {
            "runs": bat1.get("runs", 0),
            "innings": bat1.get("innings", 0),
            "average": round(bat1.get("runs", 0) / max(1, bat1.get("innings", 1)), 2),
            "strike_rate": round((bat1.get("runs", 0) / max(1, bat1.get("balls", 1))) * 100, 2) if bat1.get("balls", 0) > 0 else 0.0,
            "fours": 0,
            "sixes": 0,
        },
        "bowling": {
            "wickets": bowl1.get("wickets", 0),
            "innings": bowl1.get("innings", 0),
            "average": round(bowl1.get("runs_conceded", 0) / max(1, bowl1.get("wickets", 1)), 2),
            "economy": round(bowl1.get("runs_conceded", 0) / max(1, bowl1.get("balls_bowled", 1) / 6), 2) if bowl1.get("balls_bowled", 0) > 0 else 0.0,
            "runs_conceded": bowl1.get("runs_conceded", 0),
        },
        "matches_played": stats1.get("matches_played", 0),
    }

    
    bat2 = stats2.get("batting", {})
    bowl2 = stats2.get("bowling", {})
    player2_stats = {
        "batting": {
            "runs": bat2.get("runs", 0),
            "innings": bat2.get("innings", 0),
            "average": round(bat2.get("runs", 0) / max(1, bat2.get("innings", 1)), 2),
            "strike_rate": round((bat2.get("runs", 0) / max(1, bat2.get("balls", 1))) * 100, 2) if bat2.get("balls", 0) > 0 else 0.0,
            "fours": 0,
            "sixes": 0,
        },
        "bowling": {
            "wickets": bowl2.get("wickets", 0),
            "innings": bowl2.get("innings", 0),
            "average": round(bowl2.get("runs_conceded", 0) / max(1, bowl2.get("wickets", 1)), 2),
            "economy": round(bowl2.get("runs_conceded", 0) / max(1, bowl2.get("balls_bowled", 1) / 6), 2) if bowl2.get("balls_bowled", 0) > 0 else 0.0,
            "runs_conceded": bowl2.get("runs_conceded", 0),
        },
        "matches_played": stats2.get("matches_played", 0),
    }

    
    comparison = f"Comparing {player1} vs {player2}: "
    comparison += f"{player1} has {player1_stats['batting']['runs']} runs and {player1_stats['bowling']['wickets']} wickets. "
    comparison += f"{player2} has {player2_stats['batting']['runs']} runs and {player2_stats['bowling']['wickets']} wickets."

    return {
        "player1": player1,
        "player2": player2,
        "player1_stats": player1_stats,
        "player2_stats": player2_stats,
        "comparison": comparison,
    }


@app.post("/analytics/pace-vs-spin")
async def get_pace_vs_spin(req: PaceVsSpinRequest):
    """Return pace vs spin analytics for a specific phase."""
    phase = req.phase if req.phase in ["powerplay", "middle", "death", "all"] else "all"

    
    all_bowlers = _bowler_points_cache if _bowler_points_cache else []

    
    if phase != "all":
        all_bowlers = filter_points_by_phase(all_bowlers, phase)

    
    breakdown = aggregate_pace_vs_spin_by_phase(all_bowlers, phase if phase != "all" else "powerplay")

    
    pace_data = breakdown.get("Pace", {})
    spin_data = breakdown.get("Spin", {})

    pace_stats = {
        "wickets": pace_data.get("wickets", 0),
        "runs": pace_data.get("runs", 0),
        "balls": pace_data.get("balls", 0),
        "economy": pace_data.get("economy", 0.0),
        "dot_percentage": pace_data.get("dot_pct", 0.0),
        "wickets_per_innings": pace_data.get("wickets_per_innings", 0.0),
        "innings": pace_data.get("innings", 0),
    }

    spin_stats = {
        "wickets": spin_data.get("wickets", 0),
        "runs": spin_data.get("runs", 0),
        "balls": spin_data.get("balls", 0),
        "economy": spin_data.get("economy", 0.0),
        "dot_percentage": spin_data.get("dot_pct", 0.0),
        "wickets_per_innings": spin_data.get("wickets_per_innings", 0.0),
        "innings": spin_data.get("innings", 0),
    }

    
    top_pace_bowlers = [
        {
            "name": b["name"],
            "wickets": b["wickets"],
            "runs": b["runs"],
            "overs": b["overs"],
            "economy": b["economy"],
            "innings": b["innings"],
        }
        for b in pace_data.get("top_bowlers", [])
    ]

    top_spin_bowlers = [
        {
            "name": b["name"],
            "wickets": b["wickets"],
            "runs": b["runs"],
            "overs": b["overs"],
            "economy": b["economy"],
            "innings": b["innings"],
        }
        for b in spin_data.get("top_bowlers", [])
    ]

    return {
        "pace_stats": pace_stats,
        "spin_stats": spin_stats,
        "top_pace_bowlers": top_pace_bowlers,
        "top_spin_bowlers": top_spin_bowlers,
    }
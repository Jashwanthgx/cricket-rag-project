import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import uuid
import re
from collections import Counter
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# Configuration from .env
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
COLLECTION     = "cricket_pro"

groq_client    = Groq(api_key=GROQ_API_KEY)
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
            "text":               data["text"],
            "team1":              data.get("team1", ""),
            "team2":              data.get("team2", ""),
            "date":               data.get("date", ""),
            "venue":              data.get("venue", ""),
            "winner":             data.get("winner", ""),
            "mom":                data.get("mom", ""),
            "top_scorer":         data.get("top_scorer", "N/A"),
            "top_scorer_runs":    data.get("top_scorer_runs", 0),
            "top_bowler":         data.get("top_bowler", "N/A"),
            "top_bowler_wickets": data.get("top_bowler_wickets", 0),
            "top_bowler_economy": data.get("top_bowler_economy", 0.0),
            "all_match_batters":  data.get("all_match_batters", []),
            "all_match_bowlers":  data.get("all_match_bowlers", []),
            "chased_win":         data.get("chased_win", False),
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
    r"\bevery\b", r"\bwho won\b", r"\bsummarize\b",
    r"\bcompare\b", r"\bdifference\b", r"\bboth\b", r"\bvs\b", r"\bversus\b"
]

NAME_ALIASES = {
    "virat kohli":     "V Kohli",
    "rohit sharma":    "RG Sharma",
    "shubman gill":    "Shubman Gill",
    "babar azam":      "Babar Azam",
    "kane williamson": "KS Williamson",
}

def normalize_query(q: str) -> str:
    q_lower = q.lower()
    for alias, official in NAME_ALIASES.items():
        if alias in q_lower:
            start = q_lower.find(alias)
            q = q[:start] + official + q[start + len(alias):]
    return q

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
    has_potential_name = any(word[0].isupper() for word in words[1:] if len(word) > 0)
    has_cricket_intent = any(term in q_lower for term in cricket_terms)
    return has_potential_name and has_cricket_intent

print("\n" + "="*50)
print("Cricket RAG Assistant  | type 'quit' to exit")
print("="*50)

while True:
    query = input("\nAsk something about the matches: ").strip()
    if query.lower() in ("quit", "exit", "q"):
        break
    if not query:
        continue

    query = normalize_query(query)
    query_vector = model.encode(query).tolist()

    if is_aggregate_query(query):
        priority_results = client.query_points(
            collection_name=COLLECTION, query=query_vector, limit=5
        ).points

        all_results = client.query_points(
            collection_name=COLLECTION, query=query_vector, limit=len(documents)
        ).points

        print(f"[Aggregate Query] - Scanning all {len(all_results)} records using Llama 3 via Groq...")

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

        for r in all_results:
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

        context_text = "PRE-COMPUTED STATISTICS (Use these for totals/most/comparisons):\n"
        context_text += "\nWIN COUNTS:\n"
        for team, count in win_counts.most_common():
            context_text += f"  {team}: {count} wins\n"

        context_text += "\nMAN OF THE MATCH COUNTS:\n"
        for player, count in mom_counts.most_common(50):
            context_text += f"  {player}: {count} awards\n"

        context_text += "\nTOTAL RUNS ACROSS ALL MATCHES:\n"
        for player, runs in total_player_runs.most_common(50):
            balls = total_player_balls.get(player, 0)
            sr = round((runs / balls * 100), 2) if balls > 0 else 0
            fours = total_player_fours.get(player, 0)
            sixes = total_player_sixes.get(player, 0)
            context_text += f"  {player}: {runs} runs ({balls} balls, SR: {sr}, 4s: {fours}, 6s: {sixes})\n"

        context_text += "\nTOTAL WICKETS ACROSS ALL MATCHES:\n"
        for player, count in total_player_wickets.most_common(50):
            context_text += f"  {player}: {count} wickets total\n"

        ctx_econ = sorted(bowler_economies.items(), key=lambda x: x[1])
        context_text += "\nBEST (LOWEST) SINGLE-MATCH ECONOMY:\n"
        for player, e in ctx_econ[:50]:
            context_text += f"  {player}: {e} economy\n"

        context_text += "\nHIGHEST RUNS BY TOP SCORER:\n"
        for player, runs in sorted(scorer_runs.items(), key=lambda x: -x[1])[:50]:
            context_text += f"  {player}: {runs} runs\n"

        context_text += "\nBEST SINGLE-MATCH WICKET HAULS:\n"
        for player, wkts in sorted(bowler_wickets.items(), key=lambda x: -x[1])[:50]:
            context_text += f"  {player}: {wkts} wickets\n"

        context_text += "\nTEAMS WITH MOST WINS BATTING SECOND (CHASING):\n"
        for team, count in chasing_wins.most_common():
            context_text += f"  {team}: {count} wins while chasing\n"

        context_text += "\n─────────────────────────────────────\n"
        context_text += "\nPRIORITY MATCH RECORDS (FULL DETAIL):\n"
        for i, res in enumerate(priority_results):
            context_text += f"MATCH {i+1}: {res.payload.get('text')}\n\n"

    else:
        search_result = client.query_points(
            collection_name=COLLECTION, query=query_vector, limit=5
        ).points
        context_text = ""
        for i, res in enumerate(search_result):
            context_text += f"MATCH RECORD #{i+1}:\n{res.payload.get('text', '')}\n\n"

    if not context_text:
        print("No relevant match data found.")
        continue

    if is_aggregate_query(query):
        prompt = f"""You are a strict Cricket Statistics Assistant.
INSTRUCTIONS:
1. Answer ONLY using the DATABASE RECORDS and PRE-COMPUTED STATISTICS below.
2. For counting, comparison, or 'total' questions, ALWAYS use the PRE-COMPUTED STATISTICS section.
3. Give a direct answer first. Then supporting detail if needed.
4. Do NOT use outside knowledge.
5. IMPORTANT: For Economy Rates, a LOWER number is BETTER. For Runs and Wickets, a HIGHER number is BETTER.
6. If a player is not in the Top 50 pre-computed statistics, assume their total is lower than the 50th player.

CONTEXT:
{context_text.strip()}

Question: {query}
Answer:"""
    else:
        prompt = f"""You are a strict Cricket Statistics Assistant.
INSTRUCTIONS:
1. Answer ONLY using the DATABASE RECORDS provided below.
2. If a player or result is NOT in the records, say "I don't have that information."
3. Do NOT use outside knowledge.

RECORDS:
{context_text.strip()}

Question: {query}
Answer:"""

    print("Querying Llama 3 via Groq...")

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=700,
    )
    answer = response.choices[0].message.content.strip()

    print("\n" + "="*40)
    print("ANSWER:")
    print(answer)
    print("="*40)

client.close()
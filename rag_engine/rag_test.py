import os
import argparse
from dotenv import load_dotenv

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import uuid
import time
import re
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
    PayloadSchemaType,
)
from sentence_transformers import SentenceTransformer


from utils import (
    GROQ_MODEL,
    PHASE_KEYWORDS, STYLE_KEYWORDS, AGGREGATE_PATTERNS, NAME_ALIASES,
    MOM_PATTERNS,
    detect_phase, detect_style, detect_mom_query,
    normalize_query, is_aggregate_query, build_qdrant_filter,
    filter_points_by_phase,
    aggregate_pace_vs_spin_by_phase, pace_vs_spin_context_str,
    aggregate_mom_awards, mom_context_str,
    player_vs_player_context_str, batter_vs_bowler_context_str,
)


try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

print(f"[device] Using: {DEVICE.upper()}"
      + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else " (install torch for GPU)"))


QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
COLLECTION     = "cricket_pro_v2"

groq_client = Groq(api_key=GROQ_API_KEY)

client      = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

model       = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)


parser = argparse.ArgumentParser(description="Cricket RAG REPL")
parser.add_argument(
    "--ingest", action="store_true",
    help="Wipe and re-upload the Qdrant collection from cricket_rag_data.jsonl. "
         "Omit this flag to query an existing collection without touching the data.",
)
args = parser.parse_args()


DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cricket_rag_data.jsonl")
if not os.path.exists(DATA_PATH):
    print(f"Error: '{DATA_PATH}' not found!")
    exit(1)

documents, payloads = [], []
with open(DATA_PATH, "r") as f:
    for line in f:
        data = json.loads(line)
        documents.append(data["text"])
        payloads.append({
            "text":        data["text"],
            "match_id":    data.get("match_id", ""),
            "player_name": data.get("player_name", ""),
            "innings_no":  data.get("innings_no", 0),
            "role":        data.get("role", ""),
            "format":      data.get("format", ""),
            "phase_stats": data.get("phase_stats", {}),
            "metadata":    data.get("metadata", {}),
        })

print(f"Loaded {len(documents)} player-innings records.")


def _ensure_payload_indexes() -> None:
    """Create all required Qdrant payload indexes idempotently."""
    keyword_fields = ["role", "metadata.bowler_style", "format","player_name"]
    integer_fields = [
        "phase_stats.powerplay.balls",
        "phase_stats.middle.balls",
        "phase_stats.death.balls",
    ]
    for field in keyword_fields:
        try:
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # already exists
    for field in integer_fields:
        try:
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.INTEGER,
            )
        except Exception:
            pass  # already exists


MAX_RETRIES  = 5
UPSERT_BATCH = 250   

def _upsert_with_retry(points_batch: list, label: str = "") -> None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client.upsert(
                collection_name=COLLECTION,
                points=points_batch,
                wait=True,  
            )
            return
        except Exception as exc:
            if attempt == MAX_RETRIES:
                print(f"   x {label} failed after {MAX_RETRIES} attempts: {exc}")
                raise
            wait_sec = 2 ** attempt   
            print(f"   ! {label} attempt {attempt} failed ({exc}). Retrying in {wait_sec}s ...")
            time.sleep(wait_sec)


if args.ingest:
    
    ENCODE_BATCH = 2048 if DEVICE == "cuda" else 512

    print("Refreshing collection...")
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    
    sample_dim = model.encode(["sample"], batch_size=1).shape[1]
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=sample_dim, distance=Distance.COSINE),
    )
    _ensure_payload_indexes()

    total      = len(documents)
    uploaded   = 0
    start_time = time.time()

    print(f"Encoding + upserting {total} documents "
          f"(encode_batch={ENCODE_BATCH}, upsert_batch={UPSERT_BATCH}) ...")

    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []

        for start in range(0, total, ENCODE_BATCH):
            end        = min(start + ENCODE_BATCH, total)
            batch_docs = documents[start:end]
            batch_docs = documents[start:end]
            batch_pays = payloads[start:end]

            
            vecs = model.encode(
                batch_docs,
                batch_size=ENCODE_BATCH,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            
            vec_list = vecs.tolist()
            pts = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec_list[i],
                    payload=batch_pays[i],
                )
                for i in range(len(batch_docs))
            ]

            
            for u_start in range(0, len(pts), UPSERT_BATCH):
                u_end  = min(u_start + UPSERT_BATCH, len(pts))
                chunk  = pts[u_start:u_end]
                label  = f"docs {start + u_start}-{start + u_end}"
                futures.append(executor.submit(_upsert_with_retry, chunk, label))

            uploaded += len(batch_docs)
            elapsed   = time.time() - start_time
            rate      = uploaded / elapsed if elapsed > 0 else 0
            eta       = (total - uploaded) / rate if rate > 0 else 0
            print(f"   Encoded {uploaded}/{total} | {rate:.0f} docs/s | ETA {eta:.0f}s")

        
        for fut in as_completed(futures):
            fut.result()

    elapsed = time.time() - start_time
    print(f"\nAll {total} records ingested in {elapsed:.1f}s "
          f"({total / elapsed:.0f} docs/s overall).")

else:
    print("[query-only mode] Skipping ingestion. Use --ingest to re-upload data.")
    _ensure_payload_indexes()


def _scroll_all(scroll_filter=None) -> list:
    """
    Paginated scroll through the entire collection.
    scroll() with pagination is the correct approach for full-collection reads
    (query_points is a similarity search, not a full scan).
    """
    results = []
    offset  = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=5_000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            scroll_filter=scroll_filter,
        )
        results.extend(batch)
        if next_offset is None:
            break
        offset = next_offset
    return results



def _extract_two_players(query: str) -> tuple[str, str] | None:
    """
    Try to extract two player names from a comparison/h2h query.
    Looks for patterns like "X vs Y", "X versus Y", "X and Y", "X compared to Y".
    Returns (player1, player2) as normalised Cricsheet names, or None if not found.
    """
    patterns = [
        r"([A-Z][^\s]+(?: [A-Z][^\s]+)*)\s+(?:vs\.?|versus)\s+([A-Z][^\s]+(?: [A-Z][^\s]+)*)",
        r"([A-Z][^\s]+(?: [A-Z][^\s]+)*)\s+and\s+([A-Z][^\s]+(?: [A-Z][^\s]+)*)",
        r"([A-Z][^\s]+(?: [A-Z][^\s]+)*)\s+compared (?:to|with)\s+([A-Z][^\s]+(?: [A-Z][^\s]+)*)",
    ]
    for pat in patterns:
        m = re.search(pat, query)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None



print("\n" + "=" * 50)
print("Cricket RAG Assistant v3 | type 'quit' to exit")
print("=" * 50)

while True:
    query = input("\nAsk something about the matches: ").strip()
    if query.lower() in ("quit", "exit", "q"):
        break
    if not query:
        continue

    
    query          = normalize_query(query)
   
    query_vector   = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()
    detected_phase = detect_phase(query)
    detected_style = detect_style(query)
    is_mom_query   = detect_mom_query(query)
    
    
    detected_player = next((val for val in NAME_ALIASES.values() if val in query), None)
    
    query_filter   = build_qdrant_filter(detected_phase, detected_style, detected_player)

    if is_aggregate_query(query):
        
        all_results = _scroll_all(scroll_filter=query_filter)

        
        if detected_phase:
            all_results = filter_points_by_phase(all_results, detected_phase)

        print(f"[Aggregate Query] Scanning {len(all_results)} records via Groq ...")

        total_runs      = Counter()
        total_balls_bat = Counter()
        total_wickets   = Counter()
        runs_conceded   = Counter()
        balls_bowled    = Counter()
        batting_inns    = Counter()
        bowling_inns    = Counter()
        mom_awards      = Counter()   

        for r in all_results:
            p    = r.payload
            name = p.get("player_name", "")
            role = p.get("role", "")
            meta = p.get("metadata", {})

            if role == "batter":
                total_runs[name]      += meta.get("total_runs",  0)
                total_balls_bat[name] += meta.get("total_balls", 0)
                batting_inns[name]    += 1
            elif role == "bowler":
                total_wickets[name] += meta.get("total_wickets", 0)
                runs_conceded[name] += meta.get("total_runs",    0)
                balls_bowled[name]  += meta.get("total_balls",   0)
                bowling_inns[name]  += 1

            
            mom = meta.get("man_of_match", "")
            if mom:
                mom_awards[mom] += 1

        context_text = "PRE-COMPUTED STATISTICS (use for totals / most / comparisons):\n"

        
        if is_mom_query:
            context_text += mom_context_str(all_results, top_n=20)

        
        two_players = _extract_two_players(query)
        if two_players:
            p1, p2 = two_players
            context_text += player_vs_player_context_str(all_results, p1, p2)

        
        context_text += "\nTOTAL RUNS ACROSS ALL INNINGS:\n"
        for player, runs in total_runs.most_common(50):
            balls = total_balls_bat.get(player, 0)
            sr    = round((runs / balls * 100), 2) if balls > 0 else 0
            inns  = batting_inns.get(player, 0)
            avg   = round(runs / inns, 2) if inns > 0 else 0
            context_text += f"  {player}: {runs} runs ({inns} inns, Avg: {avg}, SR: {sr})\n"

        context_text += "\nTOTAL WICKETS ACROSS ALL INNINGS:\n"
        for player, wkts in total_wickets.most_common(50):
            rc    = runs_conceded.get(player, 0)
            bb    = balls_bowled.get(player, 0)
            overs = bb / 6 if bb > 0 else 0
            econ  = round(rc / overs, 2) if overs > 0 else 0
            inns  = bowling_inns.get(player, 0)
            avg   = round(rc / wkts, 2) if wkts > 0 else 0
            context_text += f"  {player}: {wkts} wkts ({inns} inns, Avg: {avg}, Econ: {econ})\n"

        
        if detected_phase or detected_style:
            context_text += pace_vs_spin_context_str(all_results, detected_phase)

        context_text += "\n─────────────────────────────────────\n"
        context_text += "\nPRIORITY PLAYER-INNINGS RECORDS (FULL DETAIL):\n"
        priority = client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=5,
            query_filter=query_filter,
        ).points
        for i, res in enumerate(priority):
            context_text += f"RECORD #{i + 1}: {res.payload.get('text')}\n\n"

    else:
        search_result = client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=5,
            query_filter=query_filter,
        ).points

        
        if detected_phase:
            search_result = filter_points_by_phase(search_result, detected_phase)

        context_text = ""
        for i, res in enumerate(search_result):
            context_text += f"PLAYER-INNINGS RECORD #{i + 1}:\n{res.payload.get('text', '')}\n\n"

    if not context_text.strip():
        print("No relevant match data found.")
        continue

    if is_aggregate_query(query):
        prompt = f"""You are a strict Cricket Statistics Assistant.

INSTRUCTIONS:
1. Answer ONLY using the DATABASE RECORDS and PRE-COMPUTED STATISTICS below.
2. For counting / comparison / 'total' questions, ALWAYS use the PRE-COMPUTED STATISTICS section.
3. For pace-vs-spin comparisons, use the PACE vs SPIN BREAKDOWN section.
4. For Man of the Match questions, use the MAN OF THE MATCH AWARDS section.
5. For player comparisons, use the PLAYER COMPARISON section.
6. Give a direct answer first, then supporting detail.
7. Do NOT use outside knowledge.
8. Economy rate: lower = better. Runs/Wickets: higher = better.
9. If a player is not in the Top 50, assume their total is lower than the 50th player.
10. Each record = one player's performance in one innings.

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
4. Each record = one player's performance in one innings.

RECORDS:
{context_text.strip()}

Question: {query}
Answer:"""

    print("Querying LLM via Groq ...")
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=700,
    )
    answer = response.choices[0].message.content.strip()

    print("\n" + "=" * 40)
    print("ANSWER:")
    print(answer)
    print("=" * 40)

client.close()
import json
import os

# Configuration - adjust if your folder structure differs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRIC_DIR = os.path.join(BASE_DIR, "Cric_demo")
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cricket_rag_data.jsonl")

print(f"Scanning all files in: {CRIC_DIR}")

match_summaries = []
# Get all files, no longer limiting to 100
files = [f for f in os.listdir(CRIC_DIR) if f.endswith(".json")]
total_files = len(files)

def process_innings(innings_data):
    batsmen = {}
    bowlers = {}
    total_runs = 0
    for over in innings_data.get("overs", []):
        for delivery in over.get("deliveries", []):
            total_runs += delivery["runs"]["total"]
            extras = delivery.get("extras", {})
            is_wide = "wides" in extras
            is_noball = "noballs" in extras
            
            # Batting Logic
            batter = delivery["batter"]
            if batter not in batsmen:
                batsmen[batter] = {"r": 0, "b": 0, "4s": 0, "6s": 0}
            batsmen[batter]["r"] += delivery["runs"]["batter"]
            if not is_wide:
                batsmen[batter]["b"] += 1
            if delivery["runs"]["batter"] == 4: batsmen[batter]["4s"] += 1
            if delivery["runs"]["batter"] == 6: batsmen[batter]["6s"] += 1
            
            # Bowling Logic
            bowler = delivery["bowler"]
            if bowler not in bowlers:
                bowlers[bowler] = {"r": 0, "b": 0, "w": 0}
            bowlers[bowler]["r"] += delivery["runs"]["total"]
            if not is_wide and not is_noball:
                bowlers[bowler]["b"] += 1
            if "wickets" in delivery:
                for w in delivery["wickets"]:
                    if w["kind"] not in ["run out", "retired hurt", "obstructing the field"]:
                        bowlers[bowler]["w"] += 1
    return batsmen, bowlers, total_runs

for idx, file_name in enumerate(files):
    try:
        with open(os.path.join(CRIC_DIR, file_name), "r") as file:
            data = json.load(file)

        if "innings" not in data or len(data["innings"]) < 2:
            continue

        info = data["info"]
        teams = info["teams"]
        date = info["dates"][0]
        venue = info["venue"]
        mom = info.get("player_of_match", ["Unknown"])[0]
        outcome = info.get("outcome", {})
        winner = outcome.get("winner", "No Result")

        # Process both innings
        bat1, bowl1, runs1 = process_innings(data["innings"][0])
        bat2, bowl2, runs2 = process_innings(data["innings"][1])

        # Combine stats for the text summary
        all_batters = {**bat1, **bat2}
        all_bowlers = {**bowl1, **bowl2}
        
        # Calculate Strike Rates and Economies
        enriched_batters = {}
        for b, s in all_batters.items():
            sr = round((s["r"] / s["b"]) * 100, 2) if s["b"] > 0 else 0
            enriched_batters[b] = {**s, "sr": sr}

        enriched_bowlers = {}
        for b, s in all_bowlers.items():
            overs = s["b"] / 6
            econ = round(s["r"] / overs, 2) if overs > 0 else 0.0
            enriched_bowlers[b] = {**s, "e": econ}

        # Identify match heroes for the semantic text field
        top_s = max(enriched_batters, key=lambda x: enriched_batters[x]["r"]) if enriched_batters else None
        top_b = max(enriched_bowlers, key=lambda x: (enriched_bowlers[x]["w"], -enriched_bowlers[x]["e"])) if enriched_bowlers else None

        summary_text = (
            f"Match: {teams[0]} vs {teams[1]} | Winner: {winner} | Date: {date} | Venue: {venue}\n"
            f"Scores: {teams[0]} {runs1}, {teams[1]} {runs2}\n"
            f"Top Performer: {top_s} scored {enriched_batters[top_s]['r']} runs. "
            f"{top_b} took {enriched_bowlers[top_b]['w']} wickets."
        ) if top_s and top_b else f"Match: {teams[0]} vs {teams[1]} | Winner: {winner} | Date: {date} | Venue: {venue}"

        # Flatten batter and bowler stats for easier querying
        batters_flat = []
        for player, stats in enriched_batters.items():
            batters_flat.append({
                "name": player,
                "runs": stats["r"],
                "balls": stats["b"],
                "fours": stats["4s"],
                "sixes": stats["6s"],
                "sr": stats["sr"],
                "innings": 1
            })

        bowlers_flat = []
        for player, stats in enriched_bowlers.items():
            bowlers_flat.append({
                "name": player,
                "runs_conceded": stats["r"],
                "balls_bowled": stats["b"],
                "wickets": stats["w"],
                "economy": stats["e"],
                "innings": 1
            })

        match_summaries.append({
            "text": summary_text,
            "team1": teams[0], "team2": teams[1],
            "date": date, "venue": venue, "winner": winner, "mom": mom,
            "top_scorer": top_s if top_s else "N/A",
            "top_scorer_runs": enriched_batters[top_s]["r"] if top_s else 0,
            "top_bowler": top_b if top_b else "N/A",
            "top_bowler_wickets": enriched_bowlers[top_b]["w"] if top_b else 0,
            "top_bowler_economy": enriched_bowlers[top_b]["e"] if top_b else 0.0,
            "all_match_batters": batters_flat,
            "all_match_bowlers": bowlers_flat,
            "chased_win": (winner == teams[1] and runs2 > runs1)
        })

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{total_files} matches...")

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Save to JSONL
with open(OUT_FILE, "w") as f:
    for i, summary in enumerate(match_summaries):
        summary["id"] = i + 1
        f.write(json.dumps(summary) + "\n")

print(f"\nSuccess! {len(match_summaries)} matches indexed in {OUT_FILE}")
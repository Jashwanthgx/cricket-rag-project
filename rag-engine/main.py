import json
import os

match_summaries = []

files = [f for f in os.listdir("Cric_demo") if f.endswith(".json")]
match_count = 0
processed_matches = 0

for file_name in files:
    if processed_matches == 100:
        break
    match_count += 1
    print("\nProcessing Match", match_count)
    print("File:", file_name)

    with open(os.path.join("Cric_demo", file_name), "r") as file:
        data = json.load(file)

    if "innings" not in data:
        print("Skipping match (no innings data)")
        continue

    if len(data["innings"]) < 2:
        print("Skipping match (only one innings)")
        continue

    processed_matches += 1

    date     = data["info"]["dates"][0]          
    venue    = data["info"]["venue"]
    teams    = data["info"]["teams"]
    mom      = data["info"].get("player_of_match", ["Unknown"])[0]  

    print("Match Date:", date)
    print("Venue:", venue)
    print("Teams:", " vs ".join(teams))
    print("Man of the Match:", mom)

    def process_innings(innings_data):
        batsmen = {}
        bowlers = {}
        total_runs = 0

        for over in innings_data["overs"]:
            for delivery in over["deliveries"]:
                total_runs += delivery["runs"]["total"]
                extras = delivery.get("extras", {})
                is_wide   = "wides"   in extras
                is_noball = "noballs" in extras

                batter = delivery["batter"]
                if batter not in batsmen:
                    batsmen[batter] = {"runs": 0, "balls": 0}
                batsmen[batter]["runs"] += delivery["runs"]["batter"]
                if not is_wide and not is_noball:
                    batsmen[batter]["balls"] += 1

                bowler = delivery["bowler"]
                if bowler not in bowlers:
                    bowlers[bowler] = {"runs": 0, "balls": 0, "wickets": 0}
                bowlers[bowler]["runs"] += delivery["runs"]["total"]
                if not is_wide and not is_noball:
                    bowlers[bowler]["balls"] += 1

                if "wickets" in delivery:
                    for w in delivery["wickets"]:
                        if w["kind"] not in ["run out", "retired hurt", "obstructing the field"]:
                            bowlers[bowler]["wickets"] += 1

        return batsmen, bowlers, total_runs

    bat1, bowl1, first_total_runs  = process_innings(data["innings"][0])
    bat2, bowl2, second_total_runs = process_innings(data["innings"][1])

    batsmen_stats = {}
    for batter, stats in {**bat1, **bat2}.items():
        if batter not in batsmen_stats:
            batsmen_stats[batter] = {"runs": 0, "balls": 0}
        batsmen_stats[batter]["runs"]  += stats["runs"]
        batsmen_stats[batter]["balls"] += stats["balls"]

    bowler_stats = {}
    for bowler, stats in {**bowl1, **bowl2}.items():
        if bowler not in bowler_stats:
            bowler_stats[bowler] = {"runs": 0, "balls": 0, "wickets": 0}
        bowler_stats[bowler]["runs"]    += stats["runs"]
        bowler_stats[bowler]["balls"]   += stats["balls"]
        bowler_stats[bowler]["wickets"] += stats["wickets"]

    for batter, s in batsmen_stats.items():
        s["strike_rate"] = (s["runs"] / s["balls"]) * 100 if s["balls"] > 0 else 0

    for bowler, s in bowler_stats.items():
        overs = s["balls"] / 6
        s["economy_rate"] = s["runs"] / overs if overs > 0 else 0

    top_scorer = max(
        batsmen_stats,
        key=lambda x: (batsmen_stats[x]["runs"], -batsmen_stats[x]["balls"], x)
    )
    
    top_bowler = max(
        bowler_stats,
        key=lambda x: (bowler_stats[x]["wickets"], -bowler_stats[x]["runs"], -bowler_stats[x]["economy_rate"], x)
    )

    outcome = data["info"].get("outcome", {})
    match_status = "Completed"
    if "winner" in outcome:
        winner = outcome["winner"]
    elif outcome.get("result") in ["no result", "abandoned"]:
        winner = "No Result (Match Abandoned)"
        match_status = "Abandoned"
    elif outcome.get("result") == "tie":
        winner = "Tie"
    else:
        if first_total_runs > second_total_runs:
            winner = teams[0]
        elif second_total_runs > first_total_runs:
            winner = teams[1]
        else:
            winner = "Tie/No Result"

    scorer_team = teams[0] if top_scorer in data["info"]["players"].get(teams[0], []) else teams[1]
    bowler_team = teams[0] if top_bowler in data["info"]["players"].get(teams[0], []) else teams[1]

    match_summary_text = (
        f"Match: {teams[0]} vs {teams[1]} ({match_status})\n"
        f"Date: {date} | Venue: {venue}\n"
        f"First Innings Total: {first_total_runs}\n"
        f"Second Innings Total: {second_total_runs}\n"
        f"Top Scorer: {top_scorer} (Team: {scorer_team}, {batsmen_stats[top_scorer]['runs']} runs, {batsmen_stats[top_scorer]['balls']} balls)\n"
        f"Top Bowler: {top_bowler} (Team: {bowler_team}, {bowler_stats[top_bowler]['wickets']} wickets, Econ: {bowler_stats[top_bowler]['economy_rate']:.2f})\n"
        f"Man of the Match: {mom}\n"
        f"Winner: {winner}"
    )

    match_summaries.append({
        "text":   match_summary_text,
        "team1":  teams[0],
        "team2":  teams[1],
        "date":   date,
        "venue":  venue,
        "winner": winner,
        "mom":    mom,
        "top_scorer": top_scorer,
        "top_scorer_runs": batsmen_stats[top_scorer]['runs'],
        "top_bowler": top_bowler
    })

    print(f"Processed: {teams[0]} vs {teams[1]} - Winner: {winner}")
    print("\n" + "="*40 + "\n")

print("\nTotal Files Checked:", match_count)
print("Total Matches Processed:", processed_matches)
print(f"Total Summaries Stored: {len(match_summaries)}")

output_rag_file = "cricket_rag_data.jsonl"
with open(output_rag_file, "w") as f:
    for i, summary in enumerate(match_summaries):
        rag_entry = {
            "id":                 i + 1,
            "text":               summary["text"],
            "source_match_index": i + 1,
            "team1":      summary["team1"],
            "team2":      summary["team2"],
            "date":       summary["date"],
            "venue":      summary["venue"],
            "winner":     summary["winner"],
            "mom":        summary["mom"],
            "top_scorer": summary["top_scorer"],
            "top_bowler": summary["top_bowler"]
        }
        f.write(json.dumps(rag_entry) + "\n")

print(f"\nRAG Data stored successfully in: {output_rag_file}")
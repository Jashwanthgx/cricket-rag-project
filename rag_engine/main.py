
import json
import os
import uuid
from collections import defaultdict

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCH_DATA_DIR = os.path.join(BASE_DIR, "odis_male_json")
OUT_FILE       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cricket_rag_data.jsonl")

print(f"Scanning all files in: {MATCH_DATA_DIR}")



PACE_BOWLERS = {
    # India
    "JJ Bumrah", "J Bumrah", "Mohammed Shami", "Mohammad Shami",
    "Ishant Sharma", "IS Sharma", "Shardul Thakur", "ST Thakur",
    "Mohammed Siraj", "Mohammad Siraj", "Umran Malik",
    "Prasidh Krishna", "Bhuvneshwar Kumar", "B Kumar", "BK Kumar",
    "Navdeep Saini", "Ashish Nehra", "A Nehra", "Zaheer Khan",
    "RP Singh", "S Sreesanth", "Irfan Pathan", "IK Pathan",
    "D Kulkarni", "S Kaul", "Hardik Pandya", "HH Pandya",
    "Deepak Chahar", "DL Chahar",
    # Australia
    "MA Starc", "M Starc", "PJ Cummins", "P Cummins",
    "JR Hazlewood", "J Hazlewood", "PM Siddle", "P Siddle",
    "MG Johnson", "M Johnson", "B Lee", "SR Tait", "S Tait",
    "JA Behrendorff", "J Behrendorff", "KW Richardson", "J Richardson",
    "B Stanlake", "SM Boland", "S Boland", "AJ Tye",
    # New Zealand
    "TA Boult", "T Boult", "TG Southee", "T Southee",
    "MJ Henry", "M Henry", "LH Ferguson", "L Ferguson",
    "KA Jamieson", "K Jamieson", "KD Mills", "K Mills",
    "N Wagner", "SE Milne", "S Milne", "AF Milne",
    "CH Kuggeleijn", "BJ Tickner", "B Tickner",
    # England
    "JM Anderson", "J Anderson", "SCJ Broad", "S Broad",
    "CR Woakes", "C Woakes", "BA Stokes", "B Stokes",
    "MA Wood", "M Wood", "JC Archer", "J Archer",
    "DJ Willey", "D Willey", "CJ Jordan", "C Jordan",
    "TK Curran", "T Curran", "RJW Topley", "R Topley",
    "LE Plunkett", "L Plunkett", "ST Finn", "J Finn",
    "C Overton", "S Mahmood", "LD Gregory", "L Gregory",
    "MJ Potts", "M Potts", "A Flintoff", "AR Caddick",
    "AG Wharf", "AD Mascarenhas",
    # South Africa
    "K Rabada", "KA Rabada", "DW Steyn", "D Steyn",
    "M Morkel", "VD Philander", "V Philander",
    "L Ngidi", "LT Ngidi", "A Nortje", "AN Nortje",
    "D Pretorius", "WD Parnell", "W Parnell",
    "JN Phehlukwayo", "J Phehlukwayo", "M Jansen",
    "G Coetzee", "GF Coetzee", "S Magala", "B Olivier",
    "B Hendricks", "AA Donald", "AA Noffke", "CJ Dala",
    # Pakistan
    "Wahab Riaz", "Hasan Ali", "Shaheen Afridi", "Shaheen Shah Afridi",
    "Mohammad Amir", "Junaid Khan", "Mohammad Abbas",
    "Usman Shinwari", "M Hasnain", "Mohammad Hasnain",
    "Umar Gul", "U Gul", "Naseem Shah", "Haris Rauf",
    "Aamer Yamin", "Aizaz Cheema", "Anwar Ali",
    # Sri Lanka
    "SL Malinga", "L Malinga", "RS Lakmal", "S Lakmal",
    "D Chameera", "RP Chameera", "S Fernando", "N Pradeep",
    "L Kumara", "K Rajitha", "I Udana", "P Madushan",
    "CAK Rajitha", "CBRLS Kumara",
    # West Indies
    "KAJ Roach", "K Roach", "JO Holder", "J Holder",
    "ST Gabriel", "S Gabriel", "AS Joseph", "A Joseph",
    "OC Thomas", "O Thomas", "R Shepherd", "RR Shepherd",
    "JE Taylor", "J Taylor", "AD Russell", "A Russell",
    "R Rampaul", "FH Edwards", "F Edwards",
    "SS Cottrell", "S Cottrell", "SJ Joseph", "S Joseph",
    "DM Bravo", "D Bravo",
    # Bangladesh
    "Mashrafe Mortaza", "M Mortaza", "Taskin Ahmed",
    "Mustafizur Rahman", "Rubel Hossain",
    "Abu Hider Rony", "Abu Hider", "Ebadat Hossain",
    "Shoriful Islam", "Al-Amin Hossain",
    # Afghanistan
    "Hamid Hassan", "Fazalhaq Farooqi", "Naveen ul Haq",
    "Shapoor Zadran", "Yamin Ahmadzai",
    # Ireland
    "BO Rankin", "B Rankin", "JB Little", "J Little",
    "BJ McCarthy", "MA Adair", "M Adair",
    # Zimbabwe
    "T Chatara", "TL Chatara", "BV Vitori", "BV Jarvis",
    "CB Mpofu", "D Tiripano", "B Muzarabani", "M Ngarava",
    # All-rounders classified as pace
    "A Symonds", "AJ Hall", "AM Blignaut", "AB Agarkar",
    "AJ Bichel", "AJ McKay",
}

SPIN_BOWLERS = {
    # India
    "R Ashwin", "RA Ashwin", "Ravichandran Ashwin",
    "R Jadeja", "RA Jadeja", "Ravindra Jadeja",
    "YS Chahal", "Y Chahal", "Yuzvendra Chahal",
    "Kuldeep Yadav", "K Yadav", "KC Yadav",
    "Axar Patel", "AR Patel",
    "Washington Sundar", "W Sundar",
    "Piyush Chawla", "PP Chawla",
    "AM Mishra", "A Mishra",
    # Australia
    "A Zampa", "AC Zampa", "AJ Agar", "A Agar",
    "NM Lyon", "N Lyon", "MP Swepson", "M Swepson",
    # New Zealand
    "MJ Santner", "M Santner", "IS Sodhi", "I Sodhi",
    "TCA Astle", "T Astle", "AY Somerville", "A Somerville",
    "DL Vettori", "D Vettori", "AY Patel", "Ajaz Patel",
    # England
    "MM Ali", "Moeen Ali",
    "AU Rashid", "Adil Rashid",
    "JC Leach", "J Leach", "DM Bess", "D Bess",
    "AF Giles",
    # South Africa
    "KA Maharaj", "K Maharaj", "TJ Shamsi", "T Shamsi",
    "JN Linde", "J Linde", "B Fortuin", "PJ van der Berg",
    "IT Tahir", "I Tahir",
    # Pakistan
    "Yasir Shah", "Y Shah", "Shadab Khan",
    "Imad Wasim", "I Wasim", "Mohammad Nawaz", "M Nawaz",
    "Bilal Zafar", "B Zafar", "Usman Qadir",
    "Abrar Ahmed", "Aamir Kaleem", "Abdur Rehman",
    # Sri Lanka
    "M Muralitharan", "HMRKB Herath",
    "PHKD Sandakan", "S Sandakan", "A Dananjaya",
    "DM de Silva", "D de Silva", "W Mendis",
    # West Indies
    "SP Narine", "S Narine", "RL Chase", "R Chase",
    "AJ Hosein",
    # Bangladesh
    "Shakib Al Hasan",
    "Mehidy Hasan Miraz", "Mehidy Hasan", "M Miraz",
    "Taijul Islam", "T Islam",
    "Nasum Ahmed", "N Hossain", "Nayeem Hasan",
    "Abdur Razzak",
    # Afghanistan
    "Rashid Khan", "Mujeeb Ur Rahman", "Mujeeb ur Rahman",
    "Mohammad Nabi", "M Nabi", "Qais Ahmad",
    "Zahir Khan", "Z Khan", "Ahmed Raza",
    # Ireland
    "GH Dockrell", "A Dockrell", "GJ Delany", "G Delany",
    "AR McBrine",
    # Zimbabwe
    "S Raza", "SR Raza", "R Burl", "RP Burl",
    "AG Cremer",
    # Other
    "AM Phangiso", "Phangiso",
    "J Botha",
}

_overlap = PACE_BOWLERS & SPIN_BOWLERS
if _overlap:
    print(f"WARNING: {len(_overlap)} name(s) in BOTH lists — removing from SPIN: {_overlap}")
    SPIN_BOWLERS -= _overlap

_SPIN_BOWLERS_LOWER = {n.lower() for n in SPIN_BOWLERS}
_PACE_BOWLERS_LOWER = {n.lower() for n in PACE_BOWLERS}

SPIN_PATTERNS = {
    "zampa", "rashid", "chahal", "jadeja", "ashwin", "swepson", "lyon",
    "agar", "sodhi", "santner", "somerville", "vettori", "muralitharan",
    "sandakan", "dananjaya", "kuldeep", "axar", "shakib", "mehidy",
    "miraz", "taijul", "shamsi", "maharaj", "linde", "tahir", "narine",
    "dockrell", "mujeeb", "nabi", "qais", "phangiso", "burl", "raza",
    "herath", "mendis",
}

PACE_PATTERNS = {
    "bumrah", "shami", "siraj", "starc", "cummins", "hazlewood", "boult",
    "southee", "anderson", "broad", "woakes", "rabada", "steyn", "morkel",
    "ngidi", "nortje", "malinga", "lakmal", "chameera", "roach", "holder",
    "gabriel", "mortaza", "mustafizur", "farooqi", "chatara", "muzarabani",
    "stokes", "archer", "wood", "donald", "agarkar", "milne", "caddick",
    "flintoff", "blignaut",
}



def get_phase(over_num: int) -> str:
    """
    Map Cricsheet 0-indexed over number to match phase.
    ODI: overs 0-9   → powerplay  (deliveries in overs 1-10 in 1-indexed terms)
         overs 10-39 → middle     (overs 11-40)
         overs 40-49 → death      (overs 41-50)
    """
    if over_num < 10:
        return "powerplay"
    elif over_num < 40:
        return "middle"
    else:
        return "death"


def _name_words(name: str) -> set:
    """Return the lower-cased individual words in a bowler name."""
    return set(name.lower().split())


def get_bowler_style(bowler_name: str, bowler_data: dict = None) -> str:
    """
    Classify a bowler as 'Pace', 'Spin', or 'Unknown'.

    Layer 1 — O(1) set lookup in _SPIN_BOWLERS_LOWER.
    Layer 2 — O(1) set lookup in _PACE_BOWLERS_LOWER.
    Layer 3a — Word-level spin surname pattern match.
    Layer 3b — Word-level pace surname pattern match.
    Layer 4  — Bowling-pattern heuristic (last resort, tightened thresholds).

    BUG FIX (layers 1 & 2): was O(n) loop; now O(1) set membership.
    """
    bowler_lower = bowler_name.lower().strip()
    bowler_words = _name_words(bowler_name)

    if bowler_lower in _SPIN_BOWLERS_LOWER:
        return "Spin"

    if bowler_lower in _PACE_BOWLERS_LOWER:
        return "Pace"

    if bowler_words & SPIN_PATTERNS:
        return "Spin"

    if bowler_words & PACE_PATTERNS:
        return "Pace"

    if bowler_data and bowler_name in bowler_data:
        data        = bowler_data[bowler_name]
        total_balls = data["total_balls"]
        if total_balls >= 90:        
            mid_ratio   = data["middle"]    / total_balls
            pp_ratio    = data["powerplay"] / total_balls
            death_ratio = data["death"]     / total_balls
            economy     = data["economy"]

            if mid_ratio > 0.70 and economy < 5.2:
                return "Spin"
            if (pp_ratio > 0.25 or death_ratio > 0.25) and economy >= 5.0:
                return "Pace"

    return "Unknown"



def prescan_bowlers() -> dict:
    """
    Pre-scan all JSON files to build per-bowler ball-distribution and economy.
    Used only as a last-resort fallback in get_bowler_style() layer 4.

    BUG FIX: dict is now created INSIDE this function and returned explicitly
    instead of mutating a module-level global, avoiding NameError when
    prescan_bowlers() is called before the global initialiser runs.
    BUG FIX: tracks balls (not overs) to avoid float precision drift.
    """
    print("Pre-scanning all files to extract bowler patterns...")
    files       = [f for f in os.listdir(MATCH_DATA_DIR) if f.endswith(".json")]
    total_files = len(files)

    _patterns: dict[str, dict] = defaultdict(lambda: {
        "powerplay": 0, "middle": 0, "death": 0,
        "total_balls": 0, "runs": 0, "economy": 0.0,
    })

    iter_files = (
        tqdm(files, desc="Pre-scanning", unit="file") if _TQDM_AVAILABLE
        else files
    )

    for idx, file_name in enumerate(iter_files):
        file_path = os.path.join(MATCH_DATA_DIR, file_name)
        try:
            with open(file_path, "r") as fh:
                data = json.load(fh)
        except Exception:
            continue

        if "innings" not in data:
            continue

        for innings in data["innings"]:
            for over in innings.get("overs", []):
                over_num = over.get("over", 0)
                phase    = get_phase(over_num)

                for delivery in over.get("deliveries", []):
                    bowler    = delivery.get("bowler", "")
                    if not bowler:
                        continue

                    extras    = delivery.get("extras", {})
                    is_wide   = "wides"   in extras
                    is_noball = "noballs" in extras

                    runs = (
                        delivery["runs"]["total"]
                        - extras.get("byes", 0)
                        - extras.get("legbyes", 0)
                    )

                    if not is_wide and not is_noball:
                        _patterns[bowler][phase]       += 1
                        _patterns[bowler]["total_balls"] += 1
                        _patterns[bowler]["runs"]        += runs

        if not _TQDM_AVAILABLE and (idx + 1) % 500 == 0:
            print(f"  Pre-scanned {idx + 1}/{total_files} files...")

    for bowler, p in _patterns.items():
        balls = p["total_balls"]
        if balls > 0:
            p["economy"] = p["runs"] / (balls / 6)

    print(f"Pre-scan complete. Found {len(_patterns)} unique bowlers.")
    return dict(_patterns)




def process_player_innings(deliveries: list, player_name: str, role: str) -> dict:
    """
    Aggregate per-phase stats for one player across one innings.

    BUG FIX (dot-ball logic): on a no-ball, runs ARE scored and charged to
    the bowler, but the delivery must NOT count as a dot because a run
    (the penalty run) is always added. Guard: dots only when
    not is_wide AND not is_noball AND bowler_runs == 0.
    """
    phases = {
        "powerplay": {"runs": 0, "balls": 0, "wickets": 0, "dots": 0},
        "middle":    {"runs": 0, "balls": 0, "wickets": 0, "dots": 0},
        "death":     {"runs": 0, "balls": 0, "wickets": 0, "dots": 0},
    }

    for delivery in deliveries:
        over_num  = delivery.get("over", 0)
        phase     = get_phase(over_num)
        extras    = delivery.get("extras", {})
        is_wide   = "wides"   in extras
        is_noball = "noballs" in extras

        if role == "batter":
            if delivery.get("batter") == player_name:
                phases[phase]["runs"] += delivery["runs"]["batter"]
                if not is_wide:
                    phases[phase]["balls"] += 1
                    if delivery["runs"]["batter"] == 0:
                        phases[phase]["dots"] += 1

        elif role == "bowler":
            if delivery.get("bowler") == player_name:
                total_runs  = delivery["runs"]["total"]
                bowler_runs = (
                    total_runs
                    - extras.get("byes", 0)
                    - extras.get("legbyes", 0)
                )
                phases[phase]["runs"] += bowler_runs

                if not is_wide and not is_noball:
                    phases[phase]["balls"] += 1
                    
                    if bowler_runs == 0:
                        phases[phase]["dots"] += 1

                if "wickets" in delivery:
                    for wicket in delivery["wickets"]:
                        if wicket.get("kind") not in (
                            "run out", "retired hurt", "obstructing the field"
                        ):
                            phases[phase]["wickets"] += 1

    return phases


def _extract_mom(info: dict) -> str:
    """
    Extract Man of the Match player name from the match info dict.
    Cricsheet stores this under info['player_of_match'] (list) or
    info['event']['match_number'] depending on version.
    Returns the first listed player or empty string.
    """
    
    pop = info.get("player_of_match", [])
    if isinstance(pop, list) and pop:
        return pop[0]
    if isinstance(pop, str) and pop:
        return pop

    
    outcome = info.get("outcome", {})
    if isinstance(outcome, dict):
        mom = outcome.get("player_of_match", "")
        if mom:
            return mom if isinstance(mom, str) else (mom[0] if mom else "")

    return ""


def _extract_match_outcome(info: dict) -> dict:
    """
    Extract match outcome information from the match info dict.

    Returns a dict with:
    - winner: Winning team name
    - margin: Margin of victory (runs or wickets)
    - method: How the match was won (runs, wickets, innings, DLS, etc.)
    - result: Human-readable result string
    """
    outcome = info.get("outcome", {})
    if not outcome:
        return {
            "winner": "",
            "margin": 0,
            "method": "unknown",
            "result": "No result"
        }

    winner = outcome.get("winner", "")
    by_info = outcome.get("by", {})
    method = outcome.get("method", "")

    
    margin = 0
    win_method = "unknown"

    if "runs" in by_info:
        margin = by_info["runs"]
        win_method = "runs"
    elif "wickets" in by_info:
        margin = by_info["wickets"]
        win_method = "wickets"
    elif "innings" in by_info:
        margin = by_info["innings"]
        win_method = "innings"

    
    if method and "dls" in method.lower():
        win_method = "DLS"

    
    if winner and margin > 0:
        result = f"{winner} won by {margin} {win_method}"
    elif winner:
        result = f"{winner} won"
    else:
        result = "No result"

    return {
        "winner": winner,
        "margin": margin,
        "method": win_method,
        "result": result
    }


def process_match(file_path: str, match_id: str, bowler_data: dict = None) -> list:
    """
    Parse one Cricsheet JSON match file and return player-innings records.

    BUG FIX: Man of the Match now extracted and stored in metadata for
    every player record in the match so MOM queries work correctly.
    """
    try:
        with open(file_path, "r") as fh:
            data = json.load(fh)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    if "innings" not in data or len(data["innings"]) == 0:
        return []

    info       = data["info"]
    teams      = info.get("teams", [])
    venue      = info.get("venue", "Unknown")
    date       = info.get("dates", ["Unknown"])[0] if info.get("dates") else "Unknown"
    match_type = info.get("match_type", "ODI")
    mom        = _extract_mom(info)       
    match_outcome = _extract_match_outcome(info)  

    player_innings_points = []

    for innings_idx, innings in enumerate(data["innings"]):
        innings_no   = innings_idx + 1
        batting_team = innings.get("team", teams[0] if teams else "Unknown")
        bowling_team = (
            teams[1] if len(teams) > 1 and batting_team == teams[0] else teams[0]
        )

        all_deliveries = []
        for over in innings.get("overs", []):
            over_num = over.get("over", 0)
            for delivery in over.get("deliveries", []):
                delivery["over"] = over_num
                all_deliveries.append(delivery)

        batters = {d["batter"] for d in all_deliveries if "batter" in d}
        bowlers = {d["bowler"] for d in all_deliveries if "bowler" in d}

        
        for batter in batters:
            phases      = process_player_innings(all_deliveries, batter, "batter")
            total_runs  = sum(p["runs"]  for p in phases.values())
            total_balls = sum(p["balls"] for p in phases.values())
            sr          = round((total_runs / total_balls) * 100, 2) if total_balls > 0 else 0.0
            total_dots  = sum(p["dots"]  for p in phases.values())

            semantic_text = (
                f"Player: {batter} | Team: {batting_team} vs {bowling_team} | "
                f"Role: Batter | Innings: {innings_no} | Format: {match_type} | "
                f"Date: {date} | Venue: {venue} | "
                f"Total: {total_runs} runs off {total_balls} balls (SR: {sr}) | "
                f"Powerplay (ov 1-10): {phases['powerplay']['runs']}/"
                f"{phases['powerplay']['balls']} balls | "
                f"Middle (ov 11-40): {phases['middle']['runs']}/"
                f"{phases['middle']['balls']} balls | "
                f"Death (ov 41-50): {phases['death']['runs']}/"
                f"{phases['death']['balls']} balls"
                + (f" | MOM: {mom}" if mom == batter else "")
            )

            player_innings_points.append({
                "id":          str(uuid.uuid4()),
                "match_id":    match_id,
                "player_name": batter,
                "innings_no":  innings_no,
                "role":        "batter",
                "format":      match_type,
                "phase_stats": phases,
                "metadata": {
                    "team":          batting_team,
                    "opponent":      bowling_team,
                    "venue":         venue,
                    "date":          date,
                    "total_runs":    total_runs,
                    "total_balls":   total_balls,
                    "strike_rate":   sr,
                    "total_dots":    total_dots,
                    "man_of_match":  mom if mom == batter else "",  # BUG FIX: new field
                },
                "text": semantic_text,
            })

        
        for bowler in bowlers:
            phases        = process_player_innings(all_deliveries, bowler, "bowler")
            bowler_style  = get_bowler_style(bowler, bowler_data)
            total_runs    = sum(p["runs"]    for p in phases.values())
            total_balls   = sum(p["balls"]   for p in phases.values())
            total_wickets = sum(p["wickets"] for p in phases.values())
            total_dots    = sum(p["dots"]    for p in phases.values())
            overs         = total_balls / 6
            economy       = round(total_runs / overs, 2) if overs > 0 else 0.0
            dot_pct       = round(total_dots / max(1, total_balls) * 100, 1)

            semantic_text = (
                f"Player: {bowler} | Team: {bowling_team} vs {batting_team} | "
                f"Role: Bowler | Style: {bowler_style} | "
                f"Innings: {innings_no} | Format: {match_type} | "
                f"Date: {date} | Venue: {venue} | "
                f"Figures: {total_wickets}-{total_runs} ({overs:.1f} ov, "
                f"Econ: {economy}, Dot%: {dot_pct}) | "
                f"Powerplay (ov 1-10): {phases['powerplay']['wickets']}/"
                f"{phases['powerplay']['runs']} ({phases['powerplay']['balls']} balls) | "
                f"Middle (ov 11-40): {phases['middle']['wickets']}/"
                f"{phases['middle']['runs']} ({phases['middle']['balls']} balls) | "
                f"Death (ov 41-50): {phases['death']['wickets']}/"
                f"{phases['death']['runs']} ({phases['death']['balls']} balls)"
                + (f" | MOM: {mom}" if mom == bowler else "")
            )

            player_innings_points.append({
                "id":          str(uuid.uuid4()),
                "match_id":    match_id,
                "player_name": bowler,
                "innings_no":  innings_no,
                "role":        "bowler",
                "format":      match_type,
                "phase_stats": phases,
                "metadata": {
                    "team":          bowling_team,
                    "opponent":      batting_team,
                    "venue":         venue,
                    "date":          date,
                    "bowler_style":  bowler_style,
                    "total_runs":    total_runs,
                    "total_balls":   total_balls,
                    "total_wickets": total_wickets,
                    "economy":       economy,
                    "dot_pct":       dot_pct,
                    "man_of_match":  mom if mom == bowler else "",   # BUG FIX: new field
                },
                "text": semantic_text,
            })

    
    if teams and len(teams) >= 2:
        team1, team2 = teams[0], teams[1]
        match_text = (
            f"Match: {team1} vs {team2} | Date: {date} | Venue: {venue} | "
            f"Format: {match_type} | Result: {match_outcome['result']} | "
            f"Winner: {match_outcome['winner']} | "
            f"Margin: {match_outcome['margin']} {match_outcome['method']} | "
            f"Man of the Match: {mom if mom else 'None'}"
        )

        match_record = {
            "id": str(uuid.uuid4()),
            "match_id": match_id,
            "player_name": f"{team1} vs {team2}",
            "innings_no": 0,
            "role": "match",
            "format": match_type,
            "phase_stats": {
                "powerplay": {"runs": 0, "balls": 0, "wickets": 0, "dots": 0},
                "middle": {"runs": 0, "balls": 0, "wickets": 0, "dots": 0},
                "death": {"runs": 0, "balls": 0, "wickets": 0, "dots": 0},
            },
            "metadata": {
                "team": team1,
                "opponent": team2,
                "venue": venue,
                "date": date,
                "match_winner": match_outcome["winner"],
                "win_margin": match_outcome["margin"],
                "win_method": match_outcome["method"],
                "man_of_match": mom,
                "total_runs": 0,
                "total_balls": 0,
            },
            "text": match_text,
        }
        player_innings_points.append(match_record)

    return player_innings_points




def main():
    bowler_data = prescan_bowlers()

    files       = sorted(f for f in os.listdir(MATCH_DATA_DIR) if f.endswith(".json"))
    total_files = len(files)
    print(f"Found {total_files} JSON files to process")

    all_player_innings = []
    processed_count    = 0
    error_count        = 0

    iter_files = (
        tqdm(files, desc="Processing matches", unit="match") if _TQDM_AVAILABLE
        else files
    )

    for idx, file_name in enumerate(iter_files):
        file_path = os.path.join(MATCH_DATA_DIR, file_name)
        match_id  = file_name.replace(".json", "")

        try:
            player_points = process_match(file_path, match_id, bowler_data)
            all_player_innings.extend(player_points)
            processed_count += 1

            if not _TQDM_AVAILABLE and (idx + 1) % 100 == 0:
                print(
                    f"Processed {idx + 1}/{total_files} matches "
                    f"({len(all_player_innings)} player-innings points)..."
                )

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            error_count += 1

    print(f"\nSaving {len(all_player_innings)} player-innings points to {OUT_FILE}...")
    with open(OUT_FILE, "w") as f:
        for point in all_player_innings:
            f.write(json.dumps(point) + "\n")

    print(f"\nSuccess! Processing complete:")
    print(f"   Matches processed:            {processed_count}/{total_files}")
    print(f"   Errors encountered:           {error_count}")
    print(f"   Player-innings points:        {len(all_player_innings)}")
    print(f"   Output file:                  {OUT_FILE}")

    if all_player_innings:
        batters = [p for p in all_player_innings if p["role"] == "batter"]
        bowlers = [p for p in all_player_innings if p["role"] == "bowler"]
        print(f"\nStatistics:")
        print(f"   Batter innings:   {len(batters)}")
        print(f"   Bowler innings:   {len(bowlers)}")
        print(f"   Avg players/match: {len(all_player_innings) / max(1, processed_count):.1f}")

        style_counts = defaultdict(int)
        for b in bowlers:
            style = b["metadata"].get("bowler_style", "Unknown")
            style_counts[style] += 1

        print(f"\nBowler Style Distribution:")
        for style, count in sorted(style_counts.items()):
            pct = 100 * count / max(1, len(bowlers))
            print(f"   {style}: {count} ({pct:.1f}%)")

        pace_pct = 100 * style_counts.get("Pace", 0) / max(1, len(bowlers))
        spin_pct = 100 * style_counts.get("Spin", 0) / max(1, len(bowlers))
        print()
        if pace_pct >= 35 and spin_pct <= 55:
            print("Distribution looks realistic for ODI cricket.")
        elif spin_pct > 60:
            print("WARNING: Spin still high. Review SPIN_BOWLERS or tighten heuristic.")
        elif pace_pct < 20:
            print("WARNING: Pace unexpectedly low. Review PACE_BOWLERS list.")

        
        from collections import Counter
        mom_counts = Counter()
        for p in all_player_innings:
            mom = p["metadata"].get("man_of_match", "")
            if mom:
                mom_counts[mom] += 1
        if mom_counts:
            print(f"\nTop 10 MOM award winners:")
            for name, count in mom_counts.most_common(10):
                print(f"   {name}: {count}")


if __name__ == "__main__":
    main()
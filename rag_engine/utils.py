
import os
import re
from typing import Optional


GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


PHASE_KEYWORDS: dict[str, list[str]] = {
    "powerplay": [
        "powerplay", "power play", "first 10", "first ten",
        "overs 1-10", "overs 1 to 10", "early overs", "opening overs",
    ],
    "middle": [
        "middle", "middle overs", "overs 11-40", "overs 11 to 40",
        "mid overs", "middle phase",
    ],
    "death": [
        "death", "death overs", "final overs", "last 10", "last ten",
        "overs 41-50", "overs 41 to 50", "slog overs", "end overs",
    ],
}


STYLE_KEYWORDS: dict[str, list[str]] = {
    "Pace": [
        "pace", "fast", "seamer", "quick", "speed", "express",
        "fast bowler", "seam bowler", "pacer",
    ],
    "Spin": [
        "spin", "spinner", "slow", "off spin", "leg spin",
        "wrist spin", "finger spin", "spin bowler",
    ],
}


NAME_ALIASES: dict[str, str] = {
    "virat kohli":     "V Kohli",
    "rohit sharma":    "RG Sharma",
    "shubman gill":    "Shubman Gill",
    "babar azam":      "Babar Azam",
    "kane williamson": "KS Williamson",
    "ms dhoni":        "MS Dhoni",
    "joe root":        "JE Root",
    "ben stokes":      "BA Stokes",
    "steve smith":     "SPD Smith",
    "david warner":    "DA Warner",
    "pat cummins":     "PJ Cummins",
    "mitchell starc":  "MA Starc",
    "jasprit bumrah":  "JJ Bumrah",
    "rashid khan":     "Rashid Khan",
    "shakib al hasan": "Shakib Al Hasan",
    "trent boult":     "TA Boult",
    "tim southee":     "TG Southee",
    "kagiso rabada":   "KA Rabada",
    "shaheen afridi":  "Shaheen Shah Afridi",
    "mustafizur":      "Mustafizur Rahman",
    "yuzvendra chahal": "YS Chahal",
    "kuldeep yadav":   "Kuldeep Yadav",
    "ravindra jadeja": "RA Jadeja",
    "ravichandran ashwin": "R Ashwin",
    "hardik pandya":   "HH Pandya",
    "mohammed shami":  "Mohammed Shami",
    "jasprit":         "JJ Bumrah",   # informal first-name only
}


AGGREGATE_PATTERNS: list[str] = [
    r"\bhow many\b", r"\btotal\b", r"\bcount\b", r"\blist all\b",
    r"\bmost\b", r"\bhighest\b", r"\bbest\b", r"\baverage\b",
    r"\bevery\b", r"\bwho won\b", r"\bsummarize\b",
    r"\bcompare\b", r"\bdifference\b", r"\bboth\b", r"\bvs\b", r"\bversus\b",
    r"\bstrike rate\b", r"\beconomy\b", r"\bwickets\b",
    r"\bruns\b", r"\bstats\b", r"\brecord\b",
    r"\bman of the match\b", r"\bmom\b", r"\baward\b",
    r"\bhead.to.head\b", r"\bh2h\b",
    r"\bleaderboard\b", r"\branking\b", r"\btop\b",
]


MOM_PATTERNS: list[str] = [
    r"\bman of the match\b", r"\bmom\b", r"\bplayer of the match\b",
    r"\baward\b", r"\bawards\b",
]




def detect_phase(query: str) -> Optional[str]:
    """Return the match phase mentioned in the query, or None."""
    q = query.lower()
    for phase, keywords in PHASE_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return phase
    return None


def detect_style(query: str) -> Optional[str]:
    """Return 'Pace' or 'Spin' if the query mentions a bowling style, or None."""
    q = query.lower()
    for style, keywords in STYLE_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return style
    return None


def detect_mom_query(query: str) -> bool:
    """Return True if the query is asking about Man of the Match awards."""
    q = query.lower()
    return any(re.search(p, q) for p in MOM_PATTERNS)


def normalize_query(q: str) -> str:
    """
    Replace informal player name aliases with official Cricsheet names.

    Bug fix: the previous version re-lowered the partially-mutated string
    after each replacement, which could corrupt subsequent alias lookups
    when two aliases shared a suffix.  We now keep a separate lower-case
    shadow that always reflects the current state of q.
    """
    for alias, official in NAME_ALIASES.items():
        q_lower = q.lower()         
        if alias in q_lower:
            start = q_lower.find(alias)
            q = q[:start] + official + q[start + len(alias):]
    return q


def is_aggregate_query(q: str) -> bool:
    """
    Return True if the query requires scanning all records rather than a
    simple semantic similarity search.

    Bug fix: removed hardcoded "pace"/"spin" word checks that duplicated
    STYLE_KEYWORDS logic.  Now delegates to detect_style() so there is one
    source of truth and adding a new style keyword in STYLE_KEYWORDS
    automatically works here too.
    """
    q_lower = q.lower()

    
    if any(w in q_lower for w in ["difference", "compare", "versus"]):
        return True

    
    if detect_style(q):
        return True

    
    if re.search(r"\bbetween\b", q_lower) and not any(
        p in q_lower for p in ["runs", "wickets", "stat"]
    ):
        return False

    
    if detect_mom_query(q):
        return True

    if any(re.search(p, q_lower) for p in AGGREGATE_PATTERNS):
        return True

    
    cricket_terms = ["runs", "wickets", "match", "played", "score",
                     "century", "mom", "award", "strike rate", "economy"]
    words = q.split()
    has_name   = any(w[0].isupper() for w in words[1:] if w)
    has_intent = any(term in q_lower for term in cricket_terms)
    return has_name and has_intent




def build_qdrant_filter(
    detected_phase: Optional[str],
    detected_style: Optional[str],
    player_name: Optional[str] = None, 
):
    """
    Build a Qdrant Filter from detected style and player name.
    
    By adding the player_name to the filter, we significantly speed up 
    aggregate queries by letting Qdrant pre-filter the results.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    conditions = []

    
    if detected_style:
        conditions.append(
            FieldCondition(
                key="metadata.bowler_style",
                match=MatchValue(value=detected_style),
            )
        )
    
    
    if player_name:
        conditions.append(
            FieldCondition(
                key="player_name",
                match=MatchValue(value=player_name),
            )
        )

    
    if not conditions:
        return None

    return Filter(must=conditions)


def filter_points_by_phase(points: list, phase: str) -> list:
    """
    Post-retrieval Python filter: keep only records that have at least one
    ball in the requested phase.  Call this after _scroll_all / query_points
    instead of relying on Qdrant nested-key Range filters.
    """
    filtered = []
    for r in points:
        p = r.payload if hasattr(r, "payload") else r
        phase_data = p.get("phase_stats", {}).get(phase, {})
        if phase_data.get("balls", 0) > 0:
            filtered.append(r)
    return filtered




def aggregate_pace_vs_spin_by_phase(all_points: list, phase: str) -> dict:
    """
    Aggregate pace-vs-spin statistics for a single phase across all records.

    Returns a canonical dict (same schema used by both rag_test and backend).
    Bug fix: added safe .get("dots", 0) guard so records that pre-date the
    dots field never raise a KeyError.
    """
    stats = {
        "Pace": {"wickets": 0, "runs": 0, "balls": 0, "dots": 0, "innings": 0},
        "Spin": {"wickets": 0, "runs": 0, "balls": 0, "dots": 0, "innings": 0},
    }
    bowler_phase: dict[str, dict] = {}

    for r in all_points:
        p = r.payload if hasattr(r, "payload") else r

        if p.get("role") != "bowler":
            continue
        style = p.get("metadata", {}).get("bowler_style", "")
        if style not in ("Pace", "Spin"):
            continue

        phase_data = p.get("phase_stats", {}).get(phase, {})
        balls   = phase_data.get("balls",   0)
        runs    = phase_data.get("runs",    0)
        wickets = phase_data.get("wickets", 0)
        dots    = phase_data.get("dots",    0)   
        if balls == 0:
            continue

        stats[style]["wickets"] += wickets
        stats[style]["runs"]    += runs
        stats[style]["balls"]   += balls
        stats[style]["dots"]    += dots
        stats[style]["innings"] += 1

        name = p.get("player_name", "Unknown")
        if name not in bowler_phase:
            bowler_phase[name] = {
                "style": style, "wickets": 0, "runs": 0, "balls": 0, "innings": 0,
            }
        bowler_phase[name]["wickets"] += wickets
        bowler_phase[name]["runs"]    += runs
        bowler_phase[name]["balls"]   += balls
        bowler_phase[name]["innings"] += 1

    result = {}
    for style, s in stats.items():
        overs   = round(s["balls"] / 6, 1)
        economy = round(s["runs"] / (s["balls"] / 6), 2) if s["balls"] > 0 else 0.0
        dot_pct = round(s["dots"] / max(1, s["balls"]) * 100, 1)
        wpi     = round(s["wickets"] / max(1, s["innings"]), 2)

        top = sorted(
            [(n, d) for n, d in bowler_phase.items() if d["style"] == style],
            key=lambda x: (-x[1]["wickets"],
                           x[1]["runs"] / max(1, x[1]["balls"] / 6)),
        )[:5]

        result[style] = {
            "wickets":             s["wickets"],
            "runs":                s["runs"],
            "balls":               s["balls"],
            "overs":               overs,
            "economy":             economy,
            "dot_pct":             dot_pct,
            "wickets_per_innings": wpi,
            "innings":             s["innings"],
            "top_bowlers": [
                {
                    "name":    n,
                    "wickets": d["wickets"],
                    "runs":    d["runs"],
                    "overs":   round(d["balls"] / 6, 1),
                    "economy": round(d["runs"] / max(1, d["balls"] / 6), 2),
                    "innings": d["innings"],
                }
                for n, d in top
            ],
        }

    return result


def pace_vs_spin_context_str(all_points: list, phase: Optional[str]) -> str:
    """
    Format pace-vs-spin breakdown as plain-text LLM context.

    Bug fix: phases list is now built inside the function body instead of
    using a mutable default argument (which would share state across calls).
    """
    
    phases_to_show = [phase] if phase else ["powerplay", "middle", "death"]
    ctx = ""
    for ph in phases_to_show:
        breakdown = aggregate_pace_vs_spin_by_phase(all_points, ph)
        ctx += f"\nPACE vs SPIN — {ph.upper()} OVERS:\n"
        for style in ("Pace", "Spin"):
            s = breakdown[style]
            ctx += (
                f"  {style}: {s['wickets']} wkts | {s['runs']} runs | "
                f"{s['overs']} ov | Econ {s['economy']} | "
                f"Dot% {s['dot_pct']} | Wkts/Inn {s['wickets_per_innings']} | "
                f"{s['innings']} innings\n"
            )
            if s["top_bowlers"]:
                ctx += f"  Top {style} bowlers:\n"
                for b in s["top_bowlers"]:
                    ctx += (
                        f"    {b['name']}: {b['wickets']} wkts, "
                        f"{b['runs']} runs, {b['overs']} ov, "
                        f"Econ {b['economy']}\n"
                    )
    return ctx




def aggregate_mom_awards(all_points: list) -> dict[str, int]:
    """
    Count Man of the Match awards per player across all records.
    Returns {player_name: award_count} sorted descending.
    """
    counts: dict[str, int] = {}
    for r in all_points:
        p = r.payload if hasattr(r, "payload") else r
        mom = p.get("metadata", {}).get("man_of_match", "")
        if mom:
            counts[mom] = counts.get(mom, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def mom_context_str(all_points: list, top_n: int = 20) -> str:
    """Format MOM award counts as plain-text LLM context."""
    counts = aggregate_mom_awards(all_points)
    if not counts:
        return "\nMAN OF THE MATCH AWARDS: No MOM data found in dataset.\n"
    ctx = "\nMAN OF THE MATCH AWARDS (total across all matches):\n"
    for i, (name, count) in enumerate(list(counts.items())[:top_n], 1):
        ctx += f"  {i:2d}. {name}: {count} award{'s' if count != 1 else ''}\n"
    return ctx




def player_vs_player_context_str(
    all_points: list,
    player1: str,
    player2: str,
) -> str:
    """
    Build a focused LLM context block comparing two players head-to-head.
    Aggregates batting and bowling stats for both from all_points.
    """
    def _empty():
        return {
            "batting":  {"runs": 0, "balls": 0, "innings": 0},
            "bowling":  {"wickets": 0, "runs": 0, "balls": 0, "innings": 0},
        }

    stats = {player1: _empty(), player2: _empty()}

    for r in all_points:
        p    = r.payload if hasattr(r, "payload") else r
        name = p.get("player_name", "")
        if name not in stats:
            continue
        role = p.get("role", "")
        meta = p.get("metadata", {})
        if role == "batter":
            stats[name]["batting"]["runs"]   += meta.get("total_runs",  0)
            stats[name]["batting"]["balls"]  += meta.get("total_balls", 0)
            stats[name]["batting"]["innings"] += 1
        elif role == "bowler":
            stats[name]["bowling"]["wickets"] += meta.get("total_wickets", 0)
            stats[name]["bowling"]["runs"]    += meta.get("total_runs",    0)
            stats[name]["bowling"]["balls"]   += meta.get("total_balls",   0)
            stats[name]["bowling"]["innings"] += 1

    ctx = f"\nPLAYER COMPARISON: {player1} vs {player2}\n"
    for player in (player1, player2):
        s  = stats[player]
        b  = s["batting"]
        bw = s["bowling"]
        bat_sr  = round((b["runs"] / b["balls"] * 100), 2) if b["balls"] > 0 else 0.0
        bat_avg = round(b["runs"] / max(1, b["innings"]), 2)
        bow_ov  = bw["balls"] / 6 if bw["balls"] > 0 else 0
        bow_eco = round(bw["runs"] / bow_ov, 2) if bow_ov > 0 else 0.0
        bow_avg = round(bw["runs"] / max(1, bw["wickets"]), 2)
        ctx += (
            f"\n  {player}:\n"
            f"    Batting : {b['runs']} runs | {b['innings']} inns | "
            f"Avg {bat_avg} | SR {bat_sr}\n"
            f"    Bowling : {bw['wickets']} wkts | {bw['innings']} inns | "
            f"Avg {bow_avg} | Econ {bow_eco}\n"
        )
    return ctx




def batter_vs_bowler_context_str(
    all_points: list,
    batter: str,
    bowler: str,
) -> str:
    """
    Summarise individual career stats for a batter and a bowler.

    NOTE: The Qdrant collection stores per-innings player aggregates, not
    delivery-level batter/bowler pairings.  True head-to-head delivery counts
    require re-running main.py with pairing storage enabled.  This function
    presents each player's overall career figures and is transparent about
    that limitation in the context string.
    """
    bat_stats = {"runs": 0, "balls": 0, "innings": 0}
    bow_stats = {"wickets": 0, "runs": 0, "balls": 0, "innings": 0}

    for r in all_points:
        p    = r.payload if hasattr(r, "payload") else r
        name = p.get("player_name", "")
        role = p.get("role", "")
        meta = p.get("metadata", {})

        if name == batter and role == "batter":
            bat_stats["runs"]   += meta.get("total_runs",  0)
            bat_stats["balls"]  += meta.get("total_balls", 0)
            bat_stats["innings"] += 1
        elif name == bowler and role == "bowler":
            bow_stats["wickets"] += meta.get("total_wickets", 0)
            bow_stats["runs"]    += meta.get("total_runs",    0)
            bow_stats["balls"]   += meta.get("total_balls",   0)
            bow_stats["innings"] += 1

    bat_sr  = round((bat_stats["runs"] / bat_stats["balls"] * 100), 2) if bat_stats["balls"] > 0 else 0.0
    bat_avg = round(bat_stats["runs"] / max(1, bat_stats["innings"]), 2)
    bow_ov  = bow_stats["balls"] / 6 if bow_stats["balls"] > 0 else 0
    bow_eco = round(bow_stats["runs"] / bow_ov, 2) if bow_ov > 0 else 0.0
    bow_avg = round(bow_stats["runs"] / max(1, bow_stats["wickets"]), 2)

    return (
        f"\nBATTER vs BOWLER CONTEXT ({batter} vs {bowler}):\n"
        f"  NOTE: Delivery-level pairing data not stored — these are career "
        f"figures, not head-to-head specifics.\n"
        f"  {batter} (career batting): {bat_stats['runs']} runs | "
        f"{bat_stats['innings']} inns | Avg {bat_avg} | SR {bat_sr}\n"
        f"  {bowler} (career bowling): {bow_stats['wickets']} wkts | "
        f"{bow_stats['innings']} inns | Avg {bow_avg} | Econ {bow_eco}\n"
    )
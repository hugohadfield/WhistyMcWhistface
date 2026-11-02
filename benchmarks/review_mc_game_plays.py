"""Inspect recorded 13-round Monte Carlo games at individual card level.

Public observer trace is the ONLY input to counterfactual Monte Carlo.
Actual opponent hands are deliberately not used for move quality estimates.
The high-budget comparison uses the SAME heuristic continuation as the bots;
it highlights candidate decisions for investigation, not proven mistakes.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import statistics

from alphawhist.core.FastRollout import CARD_ID, trick_winner, evaluate_actions
from alphawhist.core.InformationState import InformationState
from alphawhist.learning.recording import iter_trace_decisions


def read_games(trace_dir):
    games = []
    for path in sorted(Path(trace_dir).glob("game-*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()
                   if line.strip()]
        if not records or records[0]["type"] != "game_start" or records[-1]["type"] != "game_end":
            raise ValueError(f"Unfinished game: {path}")
        meta = records[0]["metadata"]
        names = meta["agents"]
        if len(names) != 3 or names.count("mc_aware") != 2 or names.count("random") != 1:
            raise ValueError("Expected two mc_aware and one random player")
        rounds = defaultdict(lambda: {"decisions": [], "summary": None})
        for record in records:
            if record["type"] == "decision":
                rounds[record["round"]]["decisions"].append(record)
            elif record["type"] == "round_end":
                rounds[record["round"]]["summary"] = record
        if len(rounds) != 13:
            raise ValueError("Expected thirteen completed rounds")
        for r, data in rounds.items():
            if data["summary"] is None:
                raise ValueError("Missing round score")
            cards = (7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7)[r]
            if len(data["decisions"]) != 3 * cards:
                raise ValueError("Incomplete round's card plays")
            for decision in data["decisions"]:
                if decision["chosen_card"] not in decision["legal_cards"]:
                    raise ValueError("Illegal recorded action")
        games.append({"game": len(games), "path": str(path), "metadata": meta,
                      "rounds": rounds, "end": records[-1]})
    if not games:
        raise ValueError(f"No game traces in {trace_dir}")
    return games


def card_name(card):
    return {"11": "J", "12": "Q", "13": "K", "14": "A"}.get(card[:-1], card[:-1]) + card[-1]


def full_review(games):
    output = ["# Five normal-MC games: card-by-card replay",
              "",
              "Each game has 13 rounds; P0–P2 seat labels, MC=1024 bid_aware_2 rollouts.",
              "Actions are chronological. Trick winner follows the game rules.",
              "Every action and hand snapshot comes from the player's public/private observer trace,",
              "not from opponent's hidden cards.", ""]
    for game in games:
        idx = game["game"]
        names = game["metadata"]["agents"]
        end = game["end"]
        scores = end["match_scores"]
        output.extend([f"## Game {idx+1} — " +
                       ", ".join(f"P{i}={name}" for i, name in enumerate(names)),
                       f"Final: {scores}; winning seats {end['winning_seats']}", ""])
        for r in range(13):
            data = game["rounds"][r]
            d = data["decisions"]
            info = d[0]["observation"]
            endround = data["summary"]
            output.append(f"### Round {r+1}: {info['cards_per_player']} cards, trump {info['trump']}, bids {info['bids']}")
            output.append(
                f"Tricks won {endround['tricks_won']}; round scores {endround['round_scores']}; "
                f"match scores {endround['match_scores']}")
            for trick_no, start in enumerate(range(0, len(d), 3), 1):
                trick = d[start:start+3]
                pile = [CARD_ID[v["chosen_card"]] for v in trick]
                trump = "hcsd".index(info["trump"])
                winning_position = trick_winner(pile, trump)
                winner = trick[winning_position]["player"]
                plays = " → ".join(
                    f"P{x['player']} {card_name(x['chosen_card'])}"
                    + (f" (bid {x['observation']['bids'][x['player']]}, "
                       f"won {x['observation']['tricks_won'][x['player']]}, "
                       f"{len(x['legal_cards'])} legal)" if names[x["player"]] == "mc_aware"
                       else " (random)")
                    for x in trick)
                output.append(f"- Trick {trick_no}: {plays}; **P{winner} wins**")
            output.append("")
    return "\n".join(output) + "\n"


def interesting_decisions(games, limit_each=8):
    """Stratified selection across stages, seats, bid pressure, and games."""
    rows = []
    for game in games:
        per_game = []
        for r in range(13):
            rd = game["rounds"][r]
            for decision in rd["decisions"]:
                player = decision["player"]
                if game["metadata"]["agents"][player] != "mc_aware":
                    continue
                if len(decision["legal_cards"]) <= 1:
                    continue
                s = decision["observation"]
                remaining = s["cards_per_player"] - s["trick_number"]
                needed = s["bids"][player] - s["tricks_won"][player]
                # Prioritize pivotal bids and short endgames, while selecting
                # decisions across the entire hand and every game.
                pressure = 3 if needed in (0, 1, remaining, remaining-1) else 0
                priority = (pressure + (3 if remaining <= 2 else 0)
                            + (2 if len(decision["legal_cards"]) >= 3 else 0)
                            + ((r * 17 + decision["decision_in_round"] * 3 + game["game"]) % 7) / 10)
                per_game.append((priority, decision, game["game"]))
        # Split half short (<=2 cards) and half earlier to avoid selecting
        # only two-card endgames that the heuristic already handles.
        late = [x for x in per_game if
                x[1]["observation"]["cards_per_player"]-x[1]["observation"]["trick_number"] <= 2]
        early = [x for x in per_game if
                 x[1]["observation"]["cards_per_player"]-x[1]["observation"]["trick_number"] > 2]
        take = max(1, limit_each // 2)
        selected = sorted(late, reverse=True, key=lambda x: x[0])[:take]
        selected += sorted(early, reverse=True, key=lambda x: x[0])[:limit_each-len(selected)]
        rows.extend(selected)
    return rows


def audit(games, *, samples=4096, decisions_per_game=8, seed=20261103):
    selected = interesting_decisions(games, decisions_per_game)
    analyses = []
    for position, (_, record, game_index) in enumerate(selected):
        snapshot = InformationState.from_dict(record["observation"])
        legal = tuple(record["legal_cards"])
        # Independent, shared-world MC reference with larger budget. This is
        # an imperfect model reference, not access to actual opponent hands.
        ref_seed = seed + 1000003 * game_index + 90901 * record["round"] + 379 * record["decision_in_round"]
        estimated = evaluate_actions(
            snapshot, legal, n_rollouts=samples, workers=1, batch_size=256,
            policy="bid_aware_2", seed=ref_seed)
        q = dict(zip(estimated.actions, estimated.values))
        chosen, preferred = record["chosen_card"], estimated.best_action()
        regret = q[preferred] - q[chosen]
        player = record["player"]
        bid = snapshot.bids[player]
        won = snapshot.tricks_won[player]
        needed = bid - won
        remaining = snapshot.cards_per_player-snapshot.trick_number
        notes = []
        if needed <= 0:
            notes.append("already at/above bid: avoid unnecessary tricks")
        elif needed == remaining:
            notes.append("must win every remaining trick to make bid")
        elif needed > remaining:
            notes.append("bid already impossible to make")
        elif needed == 1:
            notes.append("needs exactly one more trick")
        if not snapshot.pile:
            notes.append("leading the trick")
        else:
            notes.append("following " + card_name(snapshot.pile[0][1]))
        item = {
            "game": game_index + 1, "round": record["round"]+1,
            "decision": record["decision_in_round"], "player": player,
            "trump": snapshot.trump, "bid": bid, "won_so_far": won,
            "needed": needed, "remaining_tricks": remaining,
            "hand": list(snapshot.own_hand),
            "pile": list(snapshot.pile),
            "chosen": chosen, "reference_best": preferred,
            "values": {a: round(q[a], 3) for a in legal},
            "reference_gap": round(regret, 3),
            "final_round_tricks": record["final_tricks_won"],
            "exact_bid": record["exact_bid"],
            "actual_round_score": record["actual_round_score"],
            "actual_match_scores_after_round": record["final_match_scores"],
            "notes": notes,
        }
        analyses.append(item)
        print(f"AUDIT {position+1}/{len(selected)} "
              f"G{item['game']} R{item['round']} P{player} "
              f"played={card_name(chosen)} reference={card_name(preferred)} "
              f"gap={regret:.2f}", flush=True)
    return analyses


def report(games, analyses):
    nmc, nrand, mcmade, mcrounds = 0, 0, 0, 0
    wins = [0, 0, 0]  # MC seats split equally by role, random
    game_lines = []
    for g in games:
        names = g["metadata"]["agents"]
        scores = g["end"]["match_scores"]
        winner = g["end"]["winning_seats"]
        role_winners = [names[p] for p in winner]
        wins[0] += "mc_aware" in role_winners
        wins[1] += ("mc_aware" in role_winners and len(winner) == 2)
        wins[2] += "random" in role_winners
        game_lines.append(
            f"Game {g['game']+1}: roles {names}, scores {scores}, winning seat(s) {winner}")
        for r in range(13):
            d = g["rounds"][r]
            summary = d["summary"]
            for p, a in enumerate(names):
                if a == "mc_aware":
                    nmc += summary["tricks_won"][p]
                    mcrounds += 1
                    mcmade += (summary["bids"][p] == summary["tricks_won"][p])
                else:
                    nrand += summary["tricks_won"][p]
    top = sorted(analyses, key=lambda x: x["reference_gap"], reverse=True)
    out = [
        "# Decision review: two standard Monte Carlo players versus random",
        "",
        f"Games: {len(games)}; normal MC players use 1,024 `bid_aware_2` rollouts",
        f"Recorded card decisions: {sum(3*sum((7,6,5,4,3,2,1,2,3,4,5,6,7)) for _ in games)}",
        f"MC bids made exactly: {mcmade}/{mcrounds} ({100*mcmade/mcrounds:.1f}%)",
        f"MC trick total: {nmc}; random trick total: {nrand}",
        f"Games with at least one MC winner: {wins[0]}/{len(games)}; "
        f"random first-place appearances: {wins[2]}",
        "",
        "## Final game outcomes", *["- " + x for x in game_lines],
        "",
        "## Independent higher-budget decision checks",
        "",
        f"Audited {len(analyses)} of the MC decisions, selected across games, "
        "early rounds and two-card endgames.",
        "Reference score estimates use an independent MC draw with the **same** "
        "bid_aware_2 continuation policy, not actual hidden hands.",
        "**A disagreement is NOT proof of a mistake.** The reference itself has sampling "
        "noise, and does not model opponents' bids as evidence about hands.",
        "",
        "| G/R | Player | Bid / already won | Played | Reference preference | "
        "Estimated score loss | Actual end-round tricks |",
        "|---|---|---|---|---|---:|---|",
    ]
    for a in top[:20]:
        out.append(
            f"| {a['game']}/{a['round']} | P{a['player']} | "
            f"{a['bid']} / {a['won_so_far']} | "
            f"{card_name(a['chosen'])} | {card_name(a['reference_best'])} | "
            f"{a['reference_gap']:+.2f} | {a['final_round_tricks']} |")
    out += ["", "## The highest-disagreement decisions in context", ""]
    for a in top[:12]:
        values = ", ".join(f"{card_name(card)}={value:.2f}"
                           for card,value in sorted(a["values"].items(),
                                                   key=lambda x:x[1], reverse=True))
        out.extend([
            f"### Game {a['game']}, round {a['round']}, P{a['player']}, "
            f"card play {a['decision']}",
            f"Own hand: {', '.join(card_name(c) for c in a['hand'])}. "
            f"Trump: {a['trump']}; pile: "
            f"{[(p,card_name(c)) for p,c in a['pile']]}.",
            f"Bid {a['bid']}, had won {a['won_so_far']}, "
            f"{a['remaining_tricks']} tricks remaining "
            f"({a['needed']} still needed).",
            f"Played **{card_name(a['chosen'])}**. "
            f"Independent high-budget MC prefers **{card_name(a['reference_best'])}** "
            f"by {a['reference_gap']:.2f} expected round points.",
            f"Estimated values: {values}.",
            f"Actual round ended with {a['final_round_tricks']} tricks "
            f"for this player; {'made' if a['exact_bid'] else 'missed'} bid. "
            f"Context: {'; '.join(a['notes'])}.",
            "",
        ])
    out.extend([
        "## Interpretation",
        "",
        "These are diagnostic candidate disagreements, not a validated regret ranking.",
        "We would need repeat independent evaluations and/or full continuation "
        "under competing policies to distinguish Monte Carlo noise from policy error.",
        "Scoring 10 bonus points for exactly hitting bids makes decisions near the "
        "bid threshold particularly consequential; a high estimated disagreement "
        "may reflect errors in estimating that probability.",
        "We also need match-win objectives rather than own-round-score objectives "
        "to audit decisions based on the eventual game winner.",
        "",
        "See `all-card-plays.md` for every publicly observable action and trick winner.",
    ])
    return "\n".join(out)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-rollouts", type=int, default=4096)
    parser.add_argument("--decisions-per-game", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261103)
    args=parser.parse_args()
    games=read_games(args.traces)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_plays=full_review(games)
    (args.output_dir / "all-card-plays.md").write_text(all_plays, encoding="utf-8")
    analyses=audit(games, samples=args.audit_rollouts,
                   decisions_per_game=args.decisions_per_game, seed=args.seed)
    (args.output_dir / "candidate-decision-audit.json").write_text(
        json.dumps(analyses, indent=2), encoding="utf-8")
    markdown=report(games, analyses)
    (args.output_dir / "decision-review.md").write_text(markdown, encoding="utf-8")
    print("REVIEW_BEGIN", flush=True)
    print(markdown, flush=True)
    print("REVIEW_END", flush=True)


if __name__ == "__main__":
    main()

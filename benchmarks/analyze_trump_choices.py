"""Compare *actual* trump picks across paired, seat-swapped matches.

Input is public-trump-event CSV recorded by the full-game tournament with
--trumps-csv. No second Monte Carlo picker invocation is required, so the
experiment preserves RNG and observed full-game play.

Run:
    python -m benchmarks.analyze_trump_choices --csv trump-choices.csv
"""
import argparse
from collections import Counter, defaultdict
import csv

SUITS = "hcds"
LABEL = {"h": "hearts", "c": "clubs", "d": "diamonds", "s": "spades"}


def analyze(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or len(rows) % 26:
        raise ValueError("Expected 13 trump events for each of an even number of games")
    grouped = defaultdict(list)
    game_rounds = defaultdict(list)
    for row in rows:
        pair = int(row["pair"])
        game = int(row["game"])
        rnd = int(row["round"])
        dealer = int(row["dealer"])
        seed = int(row["deal_seed"])
        cards = int(row["cards"])
        suit, picker = row["suit"], row["picker"]
        if rnd not in range(1, 14) or cards not in range(1, 8):
            raise ValueError("Invalid round or hand size")
        if dealer != (rnd - 1) % 3 or suit not in SUITS or picker not in ("NEW", "OLD", "RANDOM"):
            raise ValueError("Invalid dealer/trump/picker")
        grouped[(pair, rnd)].append((game, dealer, seed, cards, picker, suit))
        game_rounds[game].append(rnd)

    assert all(sorted(r) == list(range(1, 14)) for r in game_rounds.values())
    by_size = defaultdict(lambda: Counter())
    new_counts, old_counts = Counter(), Counter()
    cross = Counter()
    mismatch_examples = []
    total = agreed = random_dealer_cases = 0
    for (pair, rnd), records in sorted(grouped.items()):
        if len(records) != 2:
            raise ValueError(f"Missing two seat-swap records for pair={pair}, round={rnd}")
        a,b = sorted(records)
        if a[0] != 2*pair-1 or b[0] != 2*pair:
            raise ValueError("Unexpected pairing of games")
        if a[1:4] != b[1:4]:
            raise ValueError("Paired games must share dealer, seed and hand size")
        if a[4] == "RANDOM" and b[4] == "RANDOM":
            random_dealer_cases += 1
            if a[5] != b[5]:
                raise ValueError("Original heuristic selected different trump on same dealer hand")
            continue
        if {a[4], b[4]} != {"NEW", "OLD"}:
            raise ValueError("A comparable dealer must swap NEW and OLD")
        choices = {entry[4]:entry[5] for entry in (a,b)}
        new, old = choices["NEW"], choices["OLD"]
        total += 1
        agreed += new == old
        cards = a[3]
        by_size[cards]["total"] += 1
        by_size[cards]["agree"] += new == old
        new_counts[new] += 1
        old_counts[old] += 1
        cross[(old,new)] += 1
        if new != old and len(mismatch_examples) < 12:
            mismatch_examples.append((pair, rnd, a[2], cards, old, new))

    if not total:
        raise ValueError("No paired NEW-vs-OLD trump decisions")
    print(f"TRUMP_CHOICES games={len(game_rounds)} deal_pairs={len(game_rounds)//2} "
          f"comparable_same_hand={total} same={agreed} different={total-agreed} "
          f"difference_pct={(total-agreed)*100/total:.2f} "
          f"random_dealer_cases={random_dealer_cases}")
    print("SUIT_COUNTS (on comparable identical hands, each player selected once)")
    for suit in SUITS:
        print(f"SUIT suit={suit} name={LABEL[suit]} "
              f"new={new_counts[suit]} old={old_counts[suit]}")
    print("CROSSTAB rows=legacy, columns=new (h c d s)")
    for old in SUITS:
        print(f"CROSS old={old} " + " ".join(
            f"new_{new}={cross[(old,new)]}" for new in SUITS))
    print("BY_HAND_SIZE")
    for size in sorted(by_size):
        data = by_size[size]
        print(f"SIZE cards={size} total={data['total']} same={data['agree']} "
              f"different={data['total']-data['agree']} "
              f"difference_pct={100*(data['total']-data['agree'])/data['total']:.2f}")
    print("MISMATCH_EXAMPLES")
    for pair,rnd,seed,cards,old,new in mismatch_examples:
        print(f"EXAMPLE pair={pair} round={rnd} seed={seed} "
              f"cards={cards} old={old} new={new}")
    return {
        "games":len(game_rounds),
        "comparable":total,
        "same":agreed,
        "different":total-agreed,
        "random_dealer_cases":random_dealer_cases,
        "new_suits":dict(new_counts),
        "old_suits":dict(old_counts),
        "cross":dict(cross),
        "by_size":{n:dict(counter) for n,counter in by_size.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    analyze(args.csv)


if __name__ == "__main__":
    main()

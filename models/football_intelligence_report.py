import pandas as pd

from recruitment_assistant import (
    df,
    get_player_index,
    get_similar_players,
    get_replacements,
    get_young_replacements,
    get_partners
)

from prototype_role_engine import (
    get_primary_role,
    get_role_scores
)

from squad_builder import (
    best_trio
)


def print_players(title, players):

    print(f"\n{title}")
    print("-" * len(title))

    for rank, (idx, score) in enumerate(
        players,
        start=1
    ):

        print(
            f"{rank}. "
            f"{df.iloc[idx]['Player']} "
            f"({score:.3f})"
        )


def player_report(player_name):

    player_idx = get_player_index(
        player_name
    )

    if player_idx is None:

        print(
            "Player not found."
        )

        return

    player = df.iloc[player_idx]

    print("\n")
    print("=" * 70)
    print("FOOTBALL INTELLIGENCE REPORT")
    print("=" * 70)

    print(
        f"\nPlayer: {player['Player']}"
    )

    print(
        f"Position: {player['Pos']}"
    )

    try:

        role = get_primary_role(
            player
        )

        print(
            f"Role: {role}"
        )

    except:

        pass

    print_players(
        "Top Similar Players",
        get_similar_players(
            player_idx,
            top_n=5
        )
    )

    print_players(
        "Top Replacements",
        get_replacements(
            player_idx,
            top_n=5
        )
    )

    print_players(
        "Top Compatible Partners",
        get_partners(
            player_idx,
            top_n=5
        )
    )

    print_players(
        "Young Talents (<25)",
        get_young_replacements(
            player_idx,
            top_n=5
        )
    )

    try:

        trio, chemistry = best_trio(
            player_idx
        )

        print("\nBest Unit")
        print("-" * 9)

        for idx in trio:

            p = df.iloc[idx]

            try:

                role = get_primary_role(
                    p
                )

                print(
                    f"- {p['Player']} "
                    f"| {p['Pos']} "
                    f"| {role}"
                )

            except:

                print(
                    f"- {p['Player']} "
                    f"| {p['Pos']}"
                )

        print(
            f"\nChemistry: "
            f"{chemistry:.3f}"
        )

    except Exception as e:

        print(
            "\nUnable to build trio:"
        )

        print(e)

    print("\n")
    print("=" * 70)


if __name__ == "__main__":

    player_name = input(
        "Enter player name: "
    )

    player_report(
        player_name
    )
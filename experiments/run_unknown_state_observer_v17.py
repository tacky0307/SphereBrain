from __future__ import annotations

from pathlib import Path

from concept_observer_v17 import UnknownStateObserver


def main() -> None:
    print("SphereBrain v17 — Unknown State Recurrence Observer")
    print("Phenomenon first. Recurrence second. Name later.\n")

    observer = UnknownStateObserver(similarity_threshold=0.82)

    samples = [
        ({12: 1.0, 85: 0.9, 204: 0.8, 311: 0.7}, ["空", "青い", "晴れ"]),
        ({12: 0.9, 85: 1.0, 204: 0.75, 311: 0.72}, ["空", "青い"]),
        ({44: 1.0, 105: 0.8, 500: 0.9}, ["雨", "静か", "寒い"]),
        ({13: 0.15, 85: 0.92, 204: 0.82, 311: 0.68, 12: 0.91}, ["晴れ", "暖かい"]),
        ({44: 0.95, 105: 0.82, 500: 0.88}, ["雨", "静か"]),
        ({900: 1.0, 901: 0.7, 1200: 0.8}, ["未解釈"]),
    ]

    for index, (activation, context) in enumerate(samples, start=1):
        result = observer.observe(activation, context=context)
        status = "NEW" if result.is_new else "RECURRED"
        print(
            f"observation {index:02d}: {status:<8} "
            f"state={result.state_id:<3} similarity={result.similarity:.3f} "
            f"occurrences={result.occurrences} label={result.label or '-'}"
        )

    print("\nRecurring states (2 or more observations)")
    print("-" * 72)
    for state in observer.recurrent_states(minimum_occurrences=2):
        contexts = ", ".join(
            f"{name}:{count}"
            for name, count in sorted(state.contexts.items(), key=lambda item: -item[1])
        )
        print(
            f"state={state.state_id:<3} name={state.display_name:<14} "
            f"occurrences={state.occurrences:<3} contexts=[{contexts}]"
        )

    # A label is optional and is deliberately attached only after recurrence.
    recurring = observer.recurrent_states(minimum_occurrences=3)
    if recurring:
        observer.assign_label(recurring[0].state_id, "晴天らしさ")
        print(
            f"\nHuman label attached after recurrence: "
            f"state {recurring[0].state_id} -> {recurring[0].label}"
        )

    output_path = Path("data/unknown_states_v17.json")
    observer.save(output_path)
    print(f"\nSaved observer memory: {output_path}")
    print("Unknown states remain valid even when label is null.")


if __name__ == "__main__":
    main()

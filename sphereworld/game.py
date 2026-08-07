from __future__ import annotations

import argparse
from pathlib import Path
import time

from .core_agent import CoreAgent
from .world import SphereWorld


KEY_TO_ACTION = {
    "w": "N",
    "d": "E",
    "s": "S",
    "a": "W",
    "x": "STAY",
}


def print_state(world: SphereWorld, agent: CoreAgent) -> None:
    print("\n" + "=" * 42)
    print(world.render())
    print(
        f"\nturn={world.turn}  energy={world.energy}/{world.max_energy}"
        f"  food={world.food_eaten}"
    )
    print("sense:", world.sense())
    stats = agent.core_stats()
    print(
        "Core:",
        f"experienced_nodes={stats['experienced_nodes']}",
        f"used_edges={stats['used_edges']}",
        f"edge_usage={stats['total_edge_usage']}",
        f"max_weight={stats['max_edge_weight']:.3f}",
    )


def run_teach(world: SphereWorld, agent: CoreAgent, max_steps: int) -> None:
    print("TEACH mode: W/A/S/D=move, X=stay, Q=quit")
    while world.energy > 0 and world.turn < max_steps:
        print_state(world, agent)
        senses = world.sense()

        raw = input("\nmove> ").strip().lower()
        if raw == "q":
            break
        action = KEY_TO_ACTION.get(raw)
        if action is None:
            print("W / A / S / D / X / Q を使ってください。")
            continue

        result = world.step(action)
        agent.experience(senses, action, result.outcome)
        agent.save()
        print(
            f"experience: action={action} tile={result.tile.name.lower()}"
            f" outcome={result.outcome} energy_delta={result.energy_delta:+d}"
        )

    print_state(world, agent)


def run_auto(
    world: SphereWorld,
    agent: CoreAgent,
    max_steps: int,
    delay: float,
) -> None:
    print("AUTO mode: action comes from the real Core structure.")
    while world.energy > 0 and world.turn < max_steps:
        print_state(world, agent)
        senses = world.sense()
        decision = agent.choose_action(senses)
        action = decision.action

        ranked = sorted(decision.scores.items(), key=lambda item: item[1], reverse=True)
        score_text = " ".join(f"{name}:{score:+.3f}" for name, score in ranked)
        print(f"Core decision -> {action}   [{score_text}]")

        result = world.step(action)
        agent.experience(senses, action, result.outcome)
        agent.save()
        print(
            f"result: tile={result.tile.name.lower()} outcome={result.outcome}"
            f" energy_delta={result.energy_delta:+d}"
        )
        if delay > 0:
            time.sleep(delay)

    print_state(world, agent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SphereWorld v0.1 - tiny survival world driven by SphereBrain Core"
    )
    parser.add_argument("--mode", choices=("teach", "auto"), default="teach")
    parser.add_argument("--seed", type=int, default=7, help="world random seed")
    parser.add_argument("--steps", type=int, default=100, help="maximum turns")
    parser.add_argument("--delay", type=float, default=0.25, help="auto mode delay")
    parser.add_argument(
        "--brain",
        type=Path,
        default=Path("data/sphereworld_v01/brain.json"),
        help="isolated SphereWorld Core state",
    )
    parser.add_argument(
        "--reset-core",
        action="store_true",
        help="start SphereWorld with a fresh Core",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    world = SphereWorld(size=7, seed=args.seed)
    agent = CoreAgent.load_or_create(args.brain, reset=args.reset_core)

    print("SphereWorld v0.1")
    print("World -> numeric stimulus -> real SphereBrain Core -> action -> experience")
    print(f"Core file: {args.brain}")

    if args.mode == "teach":
        run_teach(world, agent, args.steps)
    else:
        run_auto(world, agent, args.steps, args.delay)

    agent.save()
    if world.energy <= 0:
        print("\nSphere died. Core remains, so its experience can continue next run.")
    else:
        print("\nSession ended. Core experience was saved.")


if __name__ == "__main__":
    main()

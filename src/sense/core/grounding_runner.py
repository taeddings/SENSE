"""Grounding System Runner

Integrates Three-Tier Grounding with evolutionary core.
Runs closed-loop simulations and triggers evolution.
"""

from sense.grounding.tier3 import Tier3Grounding


def run_grounding_evolution(num_generations: int = 10, cycles_per_gen: int = 5):
    """Main runner for grounding + evolution loop."""
    tier3 = Tier3Grounding()

    for gen in range(num_generations):
        print(f"Generation {gen + 1}")
        feedback = tier3.run_verification_loop(cycles_per_gen)

        # Compute fitness from feedback
        if feedback:
            avg_misalignment = sum(f['verification']['misalignment_score'] for f in feedback) / len(feedback)
            print(f"Average misalignment: {avg_misalignment:.4f}")

    return tier3


if __name__ == "__main__":
    run_grounding_evolution()

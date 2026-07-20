"""Interpret iterative inverse-folding trajectories."""

from collections.abc import Mapping

from kaleprotein.core.registry import INTERPRETER_REGISTRY


@INTERPRETER_REGISTRY.register(("inverse_folding", "denoising_trajectory"))
class DenoisingTrajectoryInterpreter:
    def __init__(self, config=None):
        self.config = config

    def explain(
        self,
        output=None,
        trajectory=None,
        trajectories=None,
        **generation,
    ):
        if output is not None:
            if generation or trajectory is not None or trajectories is not None:
                raise TypeError(
                    "Pass interpretation input either as output or keyword fields, "
                    "not both."
                )
            if not isinstance(output, Mapping):
                raise TypeError("Interpretation output must be a mapping.")
            generation = dict(output)
        trajectories = trajectories or generation.get("trajectories")
        if trajectories is None:
            trajectory = trajectory or generation.get("trajectory")
            trajectories = [trajectory] if trajectory else []
        if not trajectories:
            raise ValueError(
                "No denoising trajectory is available; run an iterative sampler "
                "first."
            )

        interpreted = [_interpret_trajectory(item) for item in trajectories]
        primary = interpreted[0]
        return {
            "trajectory": primary["trajectory"],
            "trajectories": [item["trajectory"] for item in interpreted],
            "steps": primary["steps"],
            "initial_sequences": [
                sequence
                for item in interpreted
                for sequence in item["initial_sequences"]
            ],
            "final_sequences": [
                sequence
                for item in interpreted
                for sequence in item["final_sequences"]
            ],
        }


def _interpret_trajectory(trajectory):
    if not trajectory:
        raise ValueError("Denoising trajectories must contain at least one state.")
    interpreted = []
    previous = None
    for state in trajectory:
        sequences = list(state.get("sequences", []))
        changed = None
        if previous is not None and sequences:
            changed = [
                sum(first != second for first, second in zip(old, new))
                + abs(len(old) - len(new))
                for old, new in zip(previous, sequences)
            ]
        interpreted.append(
            {
                "timestep": int(state["timestep"]),
                "sequences": sequences,
                "changed_residues": changed,
            }
        )
        previous = sequences
    return {
        "trajectory": interpreted,
        "steps": len(interpreted) - 1,
        "initial_sequences": interpreted[0]["sequences"],
        "final_sequences": interpreted[-1]["sequences"],
    }


__all__ = ["DenoisingTrajectoryInterpreter"]

"""Interpret iterative inverse-folding trajectories."""

from kale_protein.core.registry import INTERPRETER_REGISTRY


@INTERPRETER_REGISTRY.register(("inverse_folding", "denoising_trajectory"))
class DenoisingTrajectoryInterpreter:
    def __init__(self, config=None):
        self.config = config

    def explain(self, predictor, data=None):
        if isinstance(predictor, dict):
            trajectory = predictor.get("trajectory")
        else:
            trajectory = getattr(predictor, "last_trajectory", None)
            if trajectory is None and data is not None and hasattr(predictor, "generate"):
                trajectory = predictor.generate(data).get("trajectory")
        if not trajectory:
            raise ValueError(
                "No denoising trajectory is available; run an iterative sampler first."
            )
        interpreted = []
        previous = None
        for state in trajectory:
            sequences = list(state.get("sequences", []))
            changed = None
            if previous is not None and sequences:
                changed = [
                    sum(a != b for a, b in zip(old, new)) + abs(len(old) - len(new))
                    for old, new in zip(previous, sequences)
                ]
            interpreted.append(
                {"timestep": int(state["timestep"]), "sequences": sequences, "changed_residues": changed}
            )
            previous = sequences
        return {
            "trajectory": interpreted,
            "steps": len(interpreted) - 1,
            "initial_sequences": interpreted[0]["sequences"],
            "final_sequences": interpreted[-1]["sequences"],
        }

from kale_protein.registry import INTERPRETER_REGISTRY
@INTERPRETER_REGISTRY.register(('inverse_folding','denoising_trajectory'))
class DenoisingTrajectoryInterpreter:
    def __init__(self, config): self.config=config
    def explain(self, predictor, data): return {'trajectory': getattr(predictor, 'last_trajectory', None), 'message': 'Trajectory returned when generated and stored.'}

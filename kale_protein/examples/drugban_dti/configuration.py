from kale_protein.auto import AutoProteinConfig


class DrugBANConfig(AutoProteinConfig):
    """Configuration for the self-contained DrugBAN model card."""

    model_type = "drugban"

    @property
    def architecture(self):
        return self.get("components", {})

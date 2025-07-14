import os
import sys
import json
import torch
import torch.nn as nn
import tempfile
import shutil
from collections import OrderedDict
import transformers

# Add the parent directory to the Python path so we can import spectrum
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectrum.spectrum import ModelModifier


# Define dummy LLM-like model
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleDict({
            "proj1": nn.Linear(768, 768),
            "proj2": nn.Linear(768, 768),
            "proj3": nn.Linear(768, 768),
        })

        # Inject a known low-rank signal into weights
        for name, layer in self.model.items():
            with torch.no_grad():
                U = torch.randn(768, 10)     # rank-10 signal
                V = torch.randn(10, 768)
                noise = 0.01 * torch.randn(768, 768)
                layer.weight.copy_(U @ V + noise)

        class DummyConfig:
            def __init__(self):
                self.rope_scaling = {'type': 'linear'}

        self.config = DummyConfig()

    def forward(self, x):
        return self.model["proj3"](self.model["proj2"](self.model["proj1"](x)))

transformers.AutoModelForCausalLM.from_pretrained = lambda *a, **kw: DummyModel()

def test_spectrum_dummy():
    os.makedirs("model_snr_results", exist_ok=True)
    model_name = "dummy/test-model"
    model_slug = model_name.replace('/', '-')
    snr_json_path = f"model_snr_results/snr_results_{model_slug}.json"

    if os.path.exists(snr_json_path):
        os.remove(snr_json_path)

    modifier = ModelModifier(model_name=model_name, top_percent=100, batch_size=1)
    selected_types = modifier.get_weight_types()
    modifier.assess_layers_snr(selected_types)
    modifier.save_snr_to_json()

    assert os.path.exists(snr_json_path), "SNR output file not created!"
    with open(snr_json_path, "r") as f:
        snr_data = json.load(f)
        assert isinstance(snr_data, dict)
        assert len(snr_data) >= 1
        for name, info in snr_data.items():
            assert "snr" in info and "type" in info
            assert isinstance(info["snr"], float)
            print(f"{name}: SNR = {info['snr']:.4f}, Type = {info['type']}")

if __name__ == "__main__":
    test_spectrum_dummy()

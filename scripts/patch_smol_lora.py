"""Keep Smol vision freezing correct after PEFT wraps the original VLM."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / 'praxis-vla/src/praxis/policies/backbones/smol/vlm.py'
s = p.read_text()
old = '    vision_model = cast(Any, vlm).model.vision_model'
if '_unwrap_peft_smol' not in s:
    assert s.count(old) == 1
    marker = 'def freeze_smol_vision(vlm: Any) -> None:'
    assert s.count(marker) == 1
    s = s.replace(marker, '''def _unwrap_peft_smol(vlm: Any) -> Any:
    get_base = getattr(vlm, "get_base_model", None)
    return get_base() if callable(get_base) else vlm


''' + marker)
    s = s.replace(old, '    vision_model = cast(Any, _unwrap_peft_smol(vlm)).model.vision_model')
    s = s.replace('    cast(Any, vlm).model.vision_model.eval()',
                  '    cast(Any, _unwrap_peft_smol(vlm)).model.vision_model.eval()')
    p.write_text(s)
p = ROOT / 'praxis-vla/src/praxis/policies/bar/backbones/smol/smol_base.py'
s = p.read_text()
if '# OAT_FREEZE_AFTER_LORA' not in s:
    marker = '    def _load_vlm(self, model_name: str, dtype_str: str) -> None:'
    assert s.count(marker) == 1
    s = s.replace(marker, '''    def __init__(self, config: Any, **kwargs) -> None:
        super().__init__(config, **kwargs)
        # OAT_FREEZE_AFTER_LORA: PEFT can add new trainable vision adapters.
        if self.config.freeze_vision_encoder:
            self._freeze_vision_encoder_modules()

''' + marker)
    p.write_text(s)
else:
    s = s.replace('def __init__(self, config: SmolPolicyConfig) -> None:',
                  'def __init__(self, config: Any, **kwargs) -> None:')
    s = s.replace('super().__init__(config)\n        # OAT_FREEZE_AFTER_LORA',
                  'super().__init__(config, **kwargs)\n        # OAT_FREEZE_AFTER_LORA')
    s = s.replace('if config.freeze_vision_encoder:', 'if self.config.freeze_vision_encoder:')
    p.write_text(s)
p = ROOT / 'praxis-vla/src/praxis/policies/backbones/smol/prefix.py'
s = p.read_text()
old = '        return cast(Any, self.vlm).model'
if old in s:
    assert s.count(old) == 1
    s = s.replace(old, '''        get_base = getattr(self.vlm, "get_base_model", None)
        owner = get_base() if callable(get_base) else self.vlm
        return cast(Any, owner).model''')
    p.write_text(s)
print('Smol PEFT vision/prefix access and post-LoRA freezing corrected')

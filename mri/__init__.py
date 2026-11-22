"""MRI sequence design and scanner simulation module (Member 3).

Components:
- gre_sequence: Spoiled GRE pulse sequence design using PyPulseq
- frequency_shift: Temperature-dependent PRF frequency shift & off-resonance mapping
- simulate_mri: Sequence-based MRI scanner simulator and pipeline stage driver
- validate_mri: Self-tests (Tests 1 through 9) validating sequence and physics
"""

from .gre_sequence import design_gre_sequence, create_system_limits
from .frequency_shift import compute_prf_shift, compute_total_off_resonance
from .simulate_mri import run_mri_simulation, run_mri_stage, simulate_frame_signal

__all__ = [
    "design_gre_sequence",
    "create_system_limits",
    "compute_prf_shift",
    "compute_total_off_resonance",
    "run_mri_simulation",
    "run_mri_stage",
    "simulate_frame_signal",
]

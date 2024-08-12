from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple
from trl import DPOConfig

@dataclass
class GPOConfig(DPOConfig):
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": ("The loss type for DPO.")})
    ropo_alpha: Optional[float] = field(default=2., metadata={"help": ("weight of ropo loss for ROPO.")})
    ropo_gamma: Optional[float] = field(default=0.1, metadata={"help": ("weight of dpo loss for ROPO.")})

    def __post_init__(self):
        if self.loss_type == "kto_pair":
            raise ValueError("Support for kto_pair has been removed in DPOTrainer. Please use KTOTrainer.")
        return super().__post_init__()
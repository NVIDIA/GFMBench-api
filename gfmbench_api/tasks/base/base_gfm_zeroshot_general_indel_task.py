# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This module does not embed third-party data download URLs.
from typing import Any, Dict, List, Optional

import numpy as np

from gfmbench_api.tasks.base.base_gfm_zero_shot_task import BaseGFMZeroShotTask


class BaseGFMZeroShotGeneralIndelTask(BaseGFMZeroShotTask):
    """
    Base class for zero-shot tasks for general indel variants.
    No fine-tuning is performed - evaluates model's pre-trained capabilities.
    
    This class handles datasets with any variant types (insertions, deletions, etc.)
    and uses only the common zero-shot metrics (no SNV-specific metrics).
    
    Subclasses must implement:
        - _create_test_dataset(): Return test dataset
        - get_task_name(): Return task name
        - _get_default_max_seq_len(): Return default max sequence length
    """

    def _get_additional_metrics(self) -> List[tuple]:
        """Return empty list (no additional metrics for general indel tasks)."""
        return []

    def _update_additional_metrics(
        self,
        additional_metrics: List[tuple],
        model: Any,
        variant_sequences: List[str],
        reference_sequences: List[str],
        labels_np: np.ndarray,
        common_outputs: Dict[str, Optional[np.ndarray]]
    ) -> None:
        """No-op for general indel tasks (no additional metrics to update)."""
        pass

    def _is_snv_only(self) -> bool:
        """Return False since this handles general indel variants, not just SNV."""
        return False


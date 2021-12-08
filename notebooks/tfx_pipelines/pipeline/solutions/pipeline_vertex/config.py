# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The pipeline configurations.
"""

import os

DATA_ROOT_DEFAULT = "gs://workshop-datasets/covertype/small"


class Config:
    """Sets configuration vars."""

    REGION = os.getenv("REGION")
    PROJECT_ID = os.getenv("PROJECT_ID")
    ARTIFACT_STORE = os.getenv("ARTIFACT_STORE")
    PIPELINE_NAME = os.getenv("PIPELINE_NAME")
    DATA_ROOT_URI = os.getenv("DATA_ROOT_URI", DATA_ROOT_DEFAULT)
    TFX_IMAGE_URI = os.getenv("TFX_IMAGE_URI")
    PIPELINE_JSON = os.getenv("PIPELINE_JSON")
    PIPELINE_ROOT = os.path.join(ARTIFACT_STORE, PIPELINE_NAME)

    TRAIN_STEPS = 2
    EVAL_STEPS = 1

    BEAM_DIRECT_PIPELINE_ARGS = [
        f"--project={PROJECT_ID}",
        f"--temp_location={os.path.join(PIPELINE_ROOT, 'beam')}",
        f"--region={REGION}",
        "--runner=DataflowRunner",
    ]

    ENABLE_CACHE = False

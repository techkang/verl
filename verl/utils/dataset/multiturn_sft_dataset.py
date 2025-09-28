# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

import json
import sys

import numpy as np
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import AutoProcessor

sys.path.insert(0, "/file_system/kangsheng/openvla-oft")
from datasets import load_dataset
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


def convert_nested_value_to_list_recursive_if_not_none(data_item):
    if isinstance(data_item, dict):
        return {k: convert_nested_value_to_list_recursive_if_not_none(v) for k, v in data_item.items() if v is not None}
    elif isinstance(data_item, list):
        return [convert_nested_value_to_list_recursive_if_not_none(elem) for elem in data_item]
    elif isinstance(data_item, np.ndarray):
        # Convert to list, then recursively process the elements of the new list
        return convert_nested_value_to_list_recursive_if_not_none(data_item.tolist())
    else:
        # Base case: item is already a primitive type (int, str, float, bool, etc.)
        return data_item


class PurePromptBuilder:
    def __init__(self, model_family: str, system_prompt: str | None = None) -> None:
        self.model_family = model_family
        self.system_prompt = system_prompt
        # TODO (siddk) =>> Can't always assume LlamaTokenizer --> FIX ME!
        self.bos, self.eos = "<s>", "</s>"

        # Get role-specific "wrap" functions
        self.wrap_human = lambda msg: f"In: {msg}\nOut: "
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def clear(self):
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        if (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        human_message = self.wrap_human(message)
        prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        # Remove prefix <bos> (if exists) because it gets auto-inserted by tokenizer!
        return self.prompt.removeprefix(self.bos).rstrip()


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(self, parquet_files: str | list[str], processor, config=None):
        # Set defaults and extract parameters from config if provided

        self.gen_cfg = GenerateConfig(
            pretrained_checkpoint="",  # 使用 HF Hub 上的基础模型进行测试
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            num_images_in_input=2,
            use_proprio=True,
            load_in_8bit=False,
            load_in_4bit=False,
            center_crop=True,
            num_open_loop_steps=NUM_ACTIONS_CHUNK,
            unnorm_key="libero_10_no_noops",
        )

        config = config or {}
        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ["right", "left_right"], (
            f"Expect pad_mode to be 'right' or 'left_right'. Got {self.pad_mode}"
        )
        self.truncation = config.get("truncation", "error")
        # for right padding
        self.max_length = config.get("max_length", 1024)
        # for left right paddding to be consistent with RL
        self.max_prompt_length = config.get("max_prompt_length", 512)
        self.max_response_length = config.get("max_response_length", 512)
        # Get messages_key from the new multiturn config structure
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        self.images_key = config.get("image_key", "images")
        self.tools_key = multiturn_config.get("tools_key", "tools")
        self.enable_thinking_key = multiturn_config.get("enable_thinking_key", "enable_thinking")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.processor: AutoProcessor = processor

        self.proprio_norm_stat = {
            "mean": [
                -0.04190658777952194,
                0.03539430722594261,
                0.8257141709327698,
                2.908308267593384,
                -0.5562185049057007,
                -0.16649018228054047,
                0.028316624462604523,
                -0.028561657294631004,
            ],
            "std": [
                0.10743364691734314,
                0.14424669742584229,
                0.2572328448295593,
                0.3441362977027893,
                1.234421730041504,
                0.3579835891723633,
                0.013308707624673843,
                0.013174631632864475,
            ],
            "max": [
                0.21031762659549713,
                0.39128610491752625,
                1.3332009315490723,
                3.6714255809783936,
                3.560650587081909,
                1.386339545249939,
                0.04160946607589722,
                0.0013633022317662835,
            ],
            "min": [
                -0.4828203022480011,
                -0.3255046010017395,
                0.445506751537323,
                1.1321442127227783,
                -3.641430377960205,
                -1.842738389968872,
                -0.0010040868073701859,
                -0.04111652821302414,
            ],
            "q01": [
                -0.3899900782108307,
                -0.2838300323486328,
                0.44795057058334353,
                1.8810229921340942,
                -2.886677579879761,
                -1.1599004411697387,
                0.002066459748893976,
                -0.04001387819647789,
            ],
            "q99": [
                0.1530261474847791,
                0.32915401458740223,
                1.2546923208236693,
                3.303542451858519,
                2.7496529006957933,
                0.6893712210655194,
                0.040048558115959164,
                -0.0017598449345678235,
            ],
        }
        self.vla_norm_stats = {
            "libero_10_no_noops": {
                "action": {
                    "mean": [
                        0.01820324920117855,
                        0.05858374014496803,
                        -0.05592384561896324,
                        0.004626928828656673,
                        0.00289608770981431,
                        -0.007673131301999092,
                        0.5457824468612671,
                    ],
                    "std": [
                        0.2825464606285095,
                        0.35904666781425476,
                        0.3673802614212036,
                        0.03770702704787254,
                        0.05429719388484955,
                        0.08725254982709885,
                        0.49815231561660767,
                    ],
                    "max": [0.9375, 0.9375, 0.9375, 0.30000001192092896, 0.29357144236564636, 0.375, 1.0],
                    "min": [
                        -0.9375,
                        -0.9375,
                        -0.9375,
                        -0.23642857372760773,
                        -0.3053571283817291,
                        -0.3675000071525574,
                        0.0,
                    ],
                    "q01": [
                        -0.6348214149475098,
                        -0.7741071581840515,
                        -0.7633928656578064,
                        -0.09749999642372131,
                        -0.14819999992847435,
                        -0.2742857038974762,
                        0.0,
                    ],
                    "q99": [
                        0.7714285850524902,
                        0.8464285731315613,
                        0.9375,
                        0.13928571343421936,
                        0.15964286029338837,
                        0.3246428668498993,
                        1.0,
                    ],
                    "mask": [True, True, True, True, True, True, False],
                },
                "proprio": {
                    "mean": [
                        -0.04190658777952194,
                        0.03539430722594261,
                        0.8257141709327698,
                        2.908308267593384,
                        -0.5562185049057007,
                        -0.16649018228054047,
                        0.028316624462604523,
                        -0.028561657294631004,
                    ],
                    "std": [
                        0.10743364691734314,
                        0.14424669742584229,
                        0.2572328448295593,
                        0.3441362977027893,
                        1.234421730041504,
                        0.3579835891723633,
                        0.013308707624673843,
                        0.013174631632864475,
                    ],
                    "max": [
                        0.21031762659549713,
                        0.39128610491752625,
                        1.3332009315490723,
                        3.6714255809783936,
                        3.560650587081909,
                        1.386339545249939,
                        0.04160946607589722,
                        0.0013633022317662835,
                    ],
                    "min": [
                        -0.4828203022480011,
                        -0.3255046010017395,
                        0.445506751537323,
                        1.1321442127227783,
                        -3.641430377960205,
                        -1.842738389968872,
                        -0.0010040868073701859,
                        -0.04111652821302414,
                    ],
                    "q01": [
                        -0.3899900782108307,
                        -0.2838300323486328,
                        0.44795057058334353,
                        1.8810229921340942,
                        -2.886677579879761,
                        -1.1599004411697387,
                        0.002066459748893976,
                        -0.04001387819647789,
                    ],
                    "q99": [
                        0.1530261474847791,
                        0.32915401458740223,
                        1.2546923208236693,
                        3.303542451858519,
                        2.7496529006957933,
                        0.6893712210655194,
                        0.040048558115959164,
                        -0.0017598449345678235,
                    ],
                },
                "num_transitions": 101469,
                "num_trajectories": 379,
            }
        }
        assert len(parquet_files) == 1
        parquet_files = parquet_files[0]
        data = load_dataset(parquet_files)
        with open(f"{parquet_files}/meta/tasks.jsonl") as f:
            tasks = f.read().strip().split("\n")
        self.tasks = [json.loads(i) for i in tasks]

        self.all_obs = data["train"]
        self.action_tokenizer = ActionTokenizer(processor.tokenizer)
        self.prompt_builder = PurePromptBuilder("openvla")
        self.max_text_len = 128

    def __len__(self):
        return len(self.all_obs)

    def __getitem__(self, item):
        obs = self.all_obs[item]
        task_index = int(obs["task_index"])
        task_label = self.tasks[task_index]["task"]

        all_images = [np.array(obs["image"]), np.array(obs["wrist_image"])]

        # Process images
        all_images = prepare_images_for_vla(all_images, self.gen_cfg)

        # Extract primary image and additional images
        primary_image = all_images.pop(0)

        # Build VLA prompt
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

        # Process primary image
        inputs = self.processor(prompt, primary_image).to(dtype=torch.bfloat16)

        # Process additional wrist images if any
        if all_images:
            all_wrist_inputs = [
                self.processor(prompt, image_wrist).to(dtype=torch.bfloat16) for image_wrist in all_images
            ]
            # Concatenate all images
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        # Process proprioception data if used
        proprio = None
        if self.gen_cfg.use_proprio:
            proprio = obs["state"]
            proprio_norm_stats = self.vla_norm_stats[self.gen_cfg.unnorm_key]["proprio"]
            obs["state"] = normalize_proprio(proprio, proprio_norm_stats)
            proprio = obs["state"]

        current_action = obs["actions"]
        next_6_actions = []
        for i in range(item + 1, item + 7):
            if i >= len(self) or self.all_obs[i]["episode_index"] != obs["episode_index"]:
                next_6_actions.extend(current_action)
            else:
                next_6_actions.extend(self.all_obs[i]["actions"])
        all_actions = current_action + next_6_actions
        all_actions_str = self.action_tokenizer(all_actions)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {task_label}?"},
            {"from": "gpt", "value": all_actions_str},
        ]
        self.prompt_builder.clear()
        for turn in conversation:
            self.prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.processor.tokenizer(self.prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        attention_mask = [1] * len(input_ids)
        loss_mask = [0] * len(input_ids)
        loss_mask[-(len(all_actions_str) + 1) :] = [1] * (len(all_actions_str) + 1)

        pad_ids = [self.processor.tokenizer.pad_token_type_id] * (self.max_text_len - len(input_ids))
        attention_pad_mask = [0] * len(pad_ids)
        position_ids = list(range(0, len(input_ids)))

        input_ids += pad_ids
        attention_mask += attention_pad_mask
        loss_mask += attention_pad_mask
        position_ids += attention_pad_mask

        result = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "position_ids": torch.tensor(position_ids),
            "responses": torch.tensor(input_ids),
            "loss_mask": torch.tensor(loss_mask),
            "response_mask": torch.tensor(loss_mask),
            "pixel_values": inputs["pixel_values"][0],
        }

        return result


if __name__ == "__main__":
    dataset = MultiTurnSFTDataset(
        ["libero_dataset"], "/file_system/common-models/moojink/openvla-7b-oft-finetuned-libero-10/"
    )
    print(dataset[0])

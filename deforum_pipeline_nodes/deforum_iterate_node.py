import math
from types import SimpleNamespace

import numexpr
import numpy as np
import pandas as pd
from deforum.animation.animation_key_frames import DeformAnimKeys
from deforum.animation.new_args import RootArgs
from deforum.datafunctions.parseq_adapter import ParseqAnimKeys
from deforum.animation.new_args import process_args, RootArgs, DeforumArgs, DeforumAnimArgs, DeforumOutputArgs, \
    ParseqArgs, LoopArgs


import os
import secrets
import torch
from PIL import Image
from deforum.datafunctions.prompt import split_weighted_subprompts
from deforum.datafunctions.seed import next_seed
from qtpy import QtCore, QtWidgets

from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pil2tensor
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.showone.pipelines.pipeline_t2v_base_pixel import tensor2vid
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.showone.showone_model import VideoGenerator
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_DEFORUM_ITERATE = get_next_opcode()


class DeforumIterateWidget(QDMNodeContentWidget):

    set_frame_signal = QtCore.Signal(int)

    def initUI(self):
        self.reset_iteration = QtWidgets.QPushButton("Reset Frame Counter")
        self.frame_counter = self.create_label("Current Frame: 0")
        self.create_button_layout([self.reset_iteration])
        self.create_main_layout(grid=1)

    def set_frame_counter(self, number):
        self.frame_counter.setText(f"Current Frame: {number}")

@register_node(OP_NODE_DEFORUM_ITERATE)
class DeforumIterateNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Deforum Iterator"
    op_code = OP_NODE_DEFORUM_ITERATE
    op_title = "Deforum Iterator"
    content_label_objname = "deforum_iterate_node"
    category = "aiNodes Deforum/DeForum"
    NodeContent_class = DeforumIterateWidget
    dim = (240, 160)

    make_dirty = True

    def __init__(self, scene):
        super().__init__(scene, inputs=[6,1], outputs=[6,1])
        self.frame_index = 0
        self.content.set_frame_signal.connect(self.content.set_frame_counter)
        self.content.reset_iteration.clicked.connect(self.reset_iteration)

    def reset_iteration(self):
        self.frame_index = 0
        self.content.set_frame_signal.emit(0)

        for node in self.scene.nodes:
            if hasattr(node, "clearOutputs"):
                node.clearOutputs()
            if hasattr(node, "markDirty"):
                node.markDirty(True)


    def evalImplementation_thread(self, index=0):
        self.content.set_frame_signal.emit(self.frame_index)

        data = self.getInputData(0)


        data = self.getInputData(0)

        root_dict = RootArgs()
        args_dict = {key: value["value"] for key, value in DeforumArgs().items()}
        anim_args_dict = {key: value["value"] for key, value in DeforumAnimArgs().items()}
        output_args_dict = {key: value["value"] for key, value in DeforumOutputArgs().items()}
        loop_args_dict = {key: value["value"] for key, value in LoopArgs().items()}
        root = SimpleNamespace(**root_dict)
        args = SimpleNamespace(**args_dict)
        anim_args = SimpleNamespace(**anim_args_dict)
        video_args = SimpleNamespace(**output_args_dict)
        parseq_args = None
        loop_args = SimpleNamespace(**loop_args_dict)
        controlnet_args = SimpleNamespace(**{"controlnet_args": "None"})

        for key, value in args.__dict__.items():
            if key in data:
                if data[key] == "":
                    val = None
                else:
                    val = data[key]
                setattr(args, key, val)

        for key, value in anim_args.__dict__.items():
            if key in data:
                if data[key] == "" and "schedule" not in key:
                    val = None
                else:
                    val = data[key]
                setattr(anim_args, key, val)

        for key, value in video_args.__dict__.items():
            if key in data:
                if data[key] == "" and "schedule" not in key:
                    val = None
                else:
                    val = data[key]
                setattr(anim_args, key, val)

        for key, value in root.__dict__.items():
            if key in data:
                if data[key] == "":
                    val = None
                else:
                    val = data[key]
                setattr(root, key, val)

        for key, value in loop_args.__dict__.items():
            if key in data:
                if data[key] == "":
                    val = None
                else:
                    val = data[key]
                setattr(loop_args, key, val)

        keys, prompt_series = get_current_keys(anim_args, args.seed, root)
        # print(f"WOULD RETURN\n{keys}\n\n{prompt_series}")

        if self.frame_index >= anim_args.max_frames:

            self.reset_iteration()

            self.frame_index = 0
            gs.should_run = False
            return [None]
        else:
            args.scale = keys.cfg_scale_schedule_series[self.frame_index]
            args.prompt = prompt_series[self.frame_index]

            args.seed = int(args.seed)
            root.seed_internal = int(root.seed_internal)
            args.seed_iter_N = int(args.seed_iter_N)
            args.seed = next_seed(args, root)


            blend_value = 0.0

            # print(frame, anim_args.diffusion_cadence, node.deforum.prompt_series)

            next_frame = self.frame_index + anim_args.diffusion_cadence
            next_prompt = None

            def generate_blend_values(distance_to_next_prompt, blend_type="linear"):
                if blend_type == "linear":
                    return [i / distance_to_next_prompt for i in range(distance_to_next_prompt + 1)]
                elif blend_type == "exponential":
                    base = 2
                    return [1 / (1 + math.exp(-8 * (i / distance_to_next_prompt - 0.5))) for i in
                            range(distance_to_next_prompt + 1)]
                else:
                    raise ValueError(f"Unknown blend type: {blend_type}")

            def find_last_prompt_change(current_index, prompt_series):
                # Step backward from the current position
                for i in range(current_index - 1, -1, -1):
                    if prompt_series[i] != prompt_series[current_index]:
                        return i
                return 0  # default to the start if no change found

            def find_next_prompt_change(current_index, prompt_series):
                # Step forward from the current position
                for i in range(current_index + 1, len(prompt_series)):
                    if prompt_series[i] != prompt_series[current_index]:
                        return i
                return len(prompt_series) - 1  # default to the end if no change found

            # Inside your main loop:

            last_prompt_change = find_last_prompt_change(self.frame_index, prompt_series)
            next_prompt_change = find_next_prompt_change(self.frame_index, prompt_series)

            distance_between_changes = next_prompt_change - last_prompt_change
            current_distance_from_last = self.frame_index - last_prompt_change

            # Generate blend values for the distance between prompt changes
            blend_values = generate_blend_values(distance_between_changes, blend_type="exponential")

            # Fetch the blend value based on the current frame's distance from the last prompt change
            blend_value = blend_values[current_distance_from_last]
            next_prompt = prompt_series[next_prompt_change]

            # print(f"Current Frame:", self.frame_index)
            # print(f"Last Prompt Change:", last_prompt_change)
            # print(f"Next Prompt Change:", next_prompt_change)
            # print(f"Distance Between Changes:", distance_between_changes)
            # print(f"Current Distance from Last Change:", current_distance_from_last)
            # print(f"Blend Value:", blend_value)
            # print(f"Blend Values:", blend_values)
            # while next_frame < anim_args.max_frames:
            #     next_prompt = prompt_series[next_frame]
            #     if next_prompt != prompt_series[self.frame_index]:
            #         distance_to_next_prompt = next_frame - self.frame_index
            #         current_distance_from_start = self.frame_index % distance_to_next_prompt
            #
            #         # Generate blend values for the current set of frames
            #         current_blend_values = generate_blend_values(distance_to_next_prompt + current_distance_from_start, blend_type="exponential")
            #
            #         # Fetch the blend value based on the current frame's distance from the starting frame
            #         blend_value = current_blend_values[current_distance_from_start]
            #
            #         print(f"Current Frame:", self.frame_index)
            #         print(f"Next Prompt Frame:", next_frame)
            #         print(f"Distance to Next Prompt:", distance_to_next_prompt)
            #         print(f"Current Distance from Start:", current_distance_from_start)
            #         print(f"Blend Value:", blend_value)
            #         print(f"Current Blend Values:", current_blend_values)
            #
            #         break  # Exit the loop once a different prompt is found
            #     next_frame += anim_args.diffusion_cadence
            # while next_frame < anim_args.max_frames:
            #     next_prompt = prompt_series[next_frame]
            #     if next_prompt != prompt_series[self.frame_index]:
            #         # Calculate blend value based on distance and frame number
            #         prompt_distance = next_frame - self.frame_index
            #         max_distance = anim_args.max_frames - self.frame_index
            #         blend_value = prompt_distance / max_distance
            #
            #         if blend_value >= 1.0:
            #             blend_value = 0.0
            #         print(f"Distance:", self.frame_index, prompt_distance, max_distance, blend_value)
            #
            #         break  # Exit the loop once a different prompt is found

                # next_frame += anim_args.diffusion_cadence




            gen_args = self.get_current_frame(args, anim_args, root, keys, self.frame_index)
            gen_args["next_prompt"] = next_prompt
            gen_args["prompt_blend"] = blend_value

            print(f"[ Deforum Iterator: {self.frame_index} / {anim_args.max_frames} ]")
            self.frame_index += 1

            return [gen_args]

    def get_current_frame(self, args, anim_args, root, keys, frame_idx):
        prompt, negative_prompt = split_weighted_subprompts(args.prompt, frame_idx, anim_args.max_frames)
        strength = keys.strength_schedule_series[frame_idx] if not frame_idx == 0 or args.use_init else 1.0

        return {"prompt":prompt,
                "negative_prompt":negative_prompt,
                "strength":strength,
                "args":args,
                "anim_args":anim_args,
                "root":root,
                "keys":keys,
                "frame_idx":frame_idx}


# prev_img, depth, mask = anim_frame_warp(prev_img, self.args, self.anim_args, keys, frame_idx, depth_model,
#                                         depth=None,
#                                         device=self.root.device, half_precision=self.root.half_precision)


def get_current_keys(anim_args, seed, root, parseq_args=None, video_args=None):

    use_parseq = False if parseq_args == None else True
    anim_args.max_frames += 1
    keys = DeformAnimKeys(anim_args, seed) if not use_parseq else ParseqAnimKeys(parseq_args, video_args)

    # Always enable pseudo-3d with parseq. No need for an extra toggle:
    # Whether it's used or not in practice is defined by the schedules
    if use_parseq:
        anim_args.flip_2d_perspective = True
        # expand prompts out to per-frame
    if use_parseq and keys.manages_prompts():
        prompt_series = keys.prompts
    else:
        prompt_series = pd.Series([np.nan for a in range(anim_args.max_frames + 1)])
        for i, prompt in root.animation_prompts.items():
            if str(i).isdigit():
                prompt_series[int(i)] = prompt
            else:
                prompt_series[int(numexpr.evaluate(i))] = prompt
        prompt_series = prompt_series.ffill().bfill()
    prompt_series = prompt_series

    return keys, prompt_series

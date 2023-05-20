import json
import os
import re
import secrets
import shutil
import sys
import time
from functools import partial
from types import SimpleNamespace

import numpy as np
from qtpy import QtCore
from qtpy import QtWidgets

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs
from custom_nodes.ainodes_engine_base_nodes.ainodes_backend import pil_image_to_pixmap
from custom_nodes.ainodes_engine_base_nodes.image_nodes.output_node import ImagePreviewNode
from custom_nodes.ainodes_engine_base_nodes.video_nodes.video_save_node import VideoOutputNode
from .deforum_basenode import DeforumBaseParamsWidget, DeforumCadenceParamsWidget, DeforumHybridParamsWidget, \
    DeforumImageInitParamsWidget, DeforumHybridScheduleParamsWidget, DeforumAnimParamsWidget, DeforumTranslationScheduleWidget, \
    DeforumColorParamsWidget, DeforumDepthParamsWidget, DeforumNoiseParamsWidget, DeforumDiffusionParamsWidget, DeforumMaskingParamsWidget, \
    DeforumVideoInitParamsWidget
from .deforum_data_no import merge_dicts

from ..deforum_helpers.render import render_animation, Root, DeforumArgs, DeforumAnimArgs, DeforumOutputArgs, \
    DeformAnimKeys, DeforumAnimPrompts, ParseqArgs, LoopArgs

OP_NODE_DEFORUM_BASE_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_CADENCE_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_HYBRID_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_IMAGE_INIT_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_HYBRID_SCHEDULE_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_ANIM_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_TRANSLATION_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_COLOR_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_DEPTH_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_NOISE_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_DIFFUSION_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_MASKING_PARAMS = get_next_opcode()
OP_NODE_DEFORUM_VIDEO_INIT_PARAMS = get_next_opcode()



#
class DeforumParamBaseNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/torch.png"
    op_code = None
    op_title = "Deforum Args Node"
    content_label_objname = "deforum_args_node"
    category = "DeForum"

    w_value = 340
    h_value = 600

    # output_socket_name = ["EXEC", "MODEL"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[6, 1], outputs=[6, 1])

    def initInnerClasses(self):
        self.content = self.content_class(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = self.w_value
        self.grNode.height = self.h_value
        self.content.setMinimumWidth(self.w_value)
        self.content.setMinimumHeight(self.h_value - 40)
        self.content.eval_signal.connect(self.evalImplementation)

    def evalImplementation_thread(self, index=0):
        self.busy = True
        input_data = self.getInputData(0)
        data = self.content.get_values()
        if input_data is not None:
            data = merge_dicts(input_data, data)
        return data

    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)
        self.setOutput(0, result)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)


@register_node(OP_NODE_DEFORUM_BASE_PARAMS)
class DeforumBaseParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Base Params"
    content_label_objname = "deforum_base_params_node"
    op_code = OP_NODE_DEFORUM_BASE_PARAMS
    content_class = DeforumBaseParamsWidget
    h_value = 735


@register_node(OP_NODE_DEFORUM_CADENCE_PARAMS)
class DeforumCadenceParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Cadence Params"
    content_label_objname = "deforum_cadence_params_node"
    op_code = OP_NODE_DEFORUM_CADENCE_PARAMS
    content_class = DeforumCadenceParamsWidget
    w_value = 400
    h_value = 300


@register_node(OP_NODE_DEFORUM_HYBRID_PARAMS)
class DeforumHybridParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Hybrid Video Params"
    content_label_objname = "deforum_hybrid_params_node"
    op_code = OP_NODE_DEFORUM_HYBRID_PARAMS
    content_class = DeforumHybridParamsWidget
    w_value = 450



@register_node(OP_NODE_DEFORUM_IMAGE_INIT_PARAMS)
class DeforumImageInitParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Image Init Params"
    content_label_objname = "deforum_image_init_params_node"
    op_code = OP_NODE_DEFORUM_IMAGE_INIT_PARAMS
    content_class = DeforumImageInitParamsWidget
    h_value = 250


@register_node(OP_NODE_DEFORUM_HYBRID_SCHEDULE_PARAMS)
class DeforumHybridSchedNode(DeforumParamBaseNode):
    op_title = "Deforum Hybrid Schedule"
    content_label_objname = "deforum_hybrid_sched_node"
    op_code = OP_NODE_DEFORUM_HYBRID_SCHEDULE_PARAMS
    content_class = DeforumHybridScheduleParamsWidget
    h_value = 300
    w_value = 450


@register_node(OP_NODE_DEFORUM_ANIM_PARAMS)
class DeforumAnimParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Anim Params"
    content_label_objname = "deforum_anim_params_node"
    op_code = OP_NODE_DEFORUM_ANIM_PARAMS
    content_class = DeforumAnimParamsWidget
    h_value = 300


@register_node(OP_NODE_DEFORUM_TRANSLATION_PARAMS)
class DeforumTranslationNode(DeforumParamBaseNode):
    op_title = "Deforum Translation"
    content_label_objname = "deforum_translation_node"
    op_code = OP_NODE_DEFORUM_TRANSLATION_PARAMS
    content_class = DeforumTranslationScheduleWidget
    h_value = 450


@register_node(OP_NODE_DEFORUM_COLOR_PARAMS)
class DeforumColorParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Color Params"
    content_label_objname = "deforum_color_params_node"
    op_code = OP_NODE_DEFORUM_COLOR_PARAMS
    content_class = DeforumColorParamsWidget
    h_value = 300

@register_node(OP_NODE_DEFORUM_DEPTH_PARAMS)
class DeforumDepthParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Depth Params"
    content_label_objname = "deforum_depth_params_node"
    op_code = OP_NODE_DEFORUM_DEPTH_PARAMS
    content_class = DeforumDepthParamsWidget
    h_value = 325


@register_node(OP_NODE_DEFORUM_NOISE_PARAMS)
class DeforumNoiseParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Noise Params"
    content_label_objname = "deforum_noise_params_node"
    op_code = OP_NODE_DEFORUM_NOISE_PARAMS
    content_class = DeforumNoiseParamsWidget
    h_value = 420 + 40


@register_node(OP_NODE_DEFORUM_DIFFUSION_PARAMS)
class DeforumDiffusionParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Diffusion Params"
    content_label_objname = "deforum_diffusion_params_node"
    op_code = OP_NODE_DEFORUM_DIFFUSION_PARAMS
    content_class = DeforumDiffusionParamsWidget
    h_value = 420


@register_node(OP_NODE_DEFORUM_MASKING_PARAMS)
class DeforumMaskingParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Masking Params"
    content_label_objname = "deforum_masking_params_node"
    op_code = OP_NODE_DEFORUM_MASKING_PARAMS
    content_class = DeforumMaskingParamsWidget
    h_value = 420 + 100


@register_node(OP_NODE_DEFORUM_VIDEO_INIT_PARAMS)
class DeforumVideoInitParamsNode(DeforumParamBaseNode):
    op_title = "Deforum Video Init Params"
    content_label_objname = "deforum_video_init_params_node"
    op_code = OP_NODE_DEFORUM_VIDEO_INIT_PARAMS
    content_class = DeforumVideoInitParamsWidget
    h_value = 420 - 100





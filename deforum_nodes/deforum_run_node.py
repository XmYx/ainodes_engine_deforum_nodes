import json
import os
import secrets
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
from qtpy import QtCore, QtWidgets
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from custom_nodes.ainodes_engine_base_nodes.ainodes_backend import pil_image_to_pixmap
from custom_nodes.ainodes_engine_base_nodes.image_nodes.output_node import ImagePreviewNode
from custom_nodes.ainodes_engine_base_nodes.video_nodes.video_save_node import VideoOutputNode
from .deforum_data_no import merge_dicts

from ..deforum_helpers.render import render_animation, Root, DeforumArgs, DeforumAnimArgs, DeforumOutputArgs, ParseqArgs, LoopArgs

OP_NODE_DEFORUM_RUN = get_next_opcode()


class DeforumRunWidget(QDMNodeContentWidget):
    progress_signal = QtCore.Signal(int)
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)
    def create_widgets(self):
        self.button = QtWidgets.QPushButton("Run")


@register_node(OP_NODE_DEFORUM_RUN)
class DeforumRunNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/torch.png"
    op_code = OP_NODE_DEFORUM_RUN
    op_title = "Deforum Runner"
    content_label_objname = "deforum_runner_node"
    category = "DeForum"

    custom_input_socket_name = ["DATA", "COND", "SAMPLER", "EXEC"]
    output_socket_name = ["IMAGE", "EXEC"]

    # output_socket_name = ["EXEC", "MODEL"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[6, 3, 5, 1], outputs=[5, 1])

    def initInnerClasses(self):
        self.content = DeforumRunWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 300
        self.grNode.height = 250
        self.content.setMinimumWidth(300)
        self.content.setMinimumHeight(250)
        self.content.eval_signal.connect(self.evalImplementation)
        #deforum_folder_name = "custom_nodes/ainodes_backend_base_nodes/ainodes_backend/deforum"
        #sys.path.extend([os.path.join(deforum_folder_name, 'scripts', 'deforum_helpers', 'src')])

    def evalImplementation_thread(self, index=0):
        self.busy = True
        try:
            data = self.getInputData(0)
        except:
            data = None

        #print("DEFORUM RUN NODE", data)

        args_dict = DeforumArgs()
        anim_args_dict = DeforumAnimArgs()
        video_args_dict = DeforumOutputArgs()
        parseq_args_dict = ParseqArgs()
        loop_args_dict = LoopArgs()
        controlnet_args = None
        root_dict = Root()

        args = SimpleNamespace(**args_dict)
        anim_args = SimpleNamespace(**anim_args_dict)
        video_args = SimpleNamespace(**video_args_dict)
        parseq_args = SimpleNamespace(**parseq_args_dict)
        loop_args = SimpleNamespace(**loop_args_dict)
        root = SimpleNamespace(**root_dict)




        def keyframeExamples():
            return '''{
            "0": "Red sphere",
            "max_f/4-5": "Cyberpunk city",
            "max_f/2-10": "Cyberpunk robot",
            "3*max_f/4-15": "Portrait of a cyberpunk soldier",
            "max_f-20": "Portrait of a cyberpunk robot soldier"
        }'''

        args_dict['animation_prompts'] = keyframeExamples()

        root.animation_prompts = json.loads(args_dict['animation_prompts'])
        if data is not None:
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

            for key, value in root.__dict__.items():
                if key in data:
                    if data[key] == "":
                        val = None
                    else:
                        val = data[key]
                    setattr(root, key, val)

        animation_prompts = root.animation_prompts
        args.timestring = time.strftime('%Y%m%d%H%M%S')
        current_arg_list = [args, anim_args, video_args, parseq_args]
        full_base_folder_path = os.path.join(os.getcwd(), "output/deforum")
        root.raw_batch_name = args.batch_name
        args.batch_name = substitute_placeholders(args.batch_name, current_arg_list, full_base_folder_path)
        args.outdir = os.path.join(full_base_folder_path, str(args.batch_name))

        if args.seed == -1 or args.seed == "-1":

            setattr(args, "seed", secrets.randbelow(999999999999999999))
            setattr(root, "raw_seed", args.seed)

        setattr(loop_args, "use_looper", True)
        try:
            test = render_animation(self, args, anim_args, video_args, parseq_args, loop_args, controlnet_args, animation_prompts, root, callback=self.handle_callback)
        except Exception as e:
            print(e)
            pass
        return True

    def handle_callback(self, image):
        pixmap = pil_image_to_pixmap(image)
        for node in self.getOutputs(0):
            if isinstance(node, ImagePreviewNode):
                node.content.preview_signal.emit(pixmap)
            elif isinstance(node, VideoOutputNode):
                frame = np.array(image)
                node.content.video.add_frame(frame, dump=node.content.dump_at.value())

    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)
        return
        self.markDirty(False)
        self.setOutput(0, result)
        pass
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)

    def onInputChanged(self, socket=None):
        pass




def get_os():
    import platform
    return {"Windows": "Windows", "Linux": "Linux", "Darwin": "Mac"}.get(platform.system(), "Unknown")

def custom_placeholder_format(value_dict, placeholder_match):
    key = placeholder_match.group(1).lower()
    value = value_dict.get(key, key) or "_"
    if isinstance(value, dict) and value:
        first_key = list(value.keys())[0]
        value = str(value[first_key][0]) if isinstance(value[first_key], list) and value[first_key] else str(value[first_key])
    return str(value)[:50]

def test_long_path_support(base_folder_path):
    long_folder_name = 'A' * 300
    long_path = os.path.join(base_folder_path, long_folder_name)
    try:
        os.makedirs(long_path)
        shutil.rmtree(long_path)
        return True
    except OSError:
        return False
def get_max_path_length(base_folder_path):
    if get_os() == 'Windows':
        return (32767 if test_long_path_support(base_folder_path) else 260) - len(base_folder_path) - 1
    return 4096 - len(base_folder_path) - 1

def substitute_placeholders(template, arg_list, base_folder_path):
    import re
    # Find and update timestring values if resume_from_timestring is True
    resume_from_timestring = next((arg_obj.resume_from_timestring for arg_obj in arg_list if hasattr(arg_obj, 'resume_from_timestring')), False)
    resume_timestring = next((arg_obj.resume_timestring for arg_obj in arg_list if hasattr(arg_obj, 'resume_timestring')), None)

    if resume_from_timestring and resume_timestring:
        for arg_obj in arg_list:
            if hasattr(arg_obj, 'timestring'):
                arg_obj.timestring = resume_timestring

    max_length = get_max_path_length(base_folder_path)
    values = {attr.lower(): getattr(arg_obj, attr)
              for arg_obj in arg_list
              for attr in dir(arg_obj) if not callable(getattr(arg_obj, attr)) and not attr.startswith('__')}
    formatted_string = re.sub(r"{(\w+)}", lambda m: custom_placeholder_format(values, m), template)
    formatted_string = re.sub(r'[<>:"/\\|?*\s,]', '_', formatted_string)
    return formatted_string[:max_length]
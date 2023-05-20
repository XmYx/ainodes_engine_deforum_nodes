import numpy as np
from qtpy import QtCore, QtWidgets
from ainodes_frontend.base import register_node, get_next_opcode, handle_ainodes_exception
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from custom_nodes.ainodes_engine_base_nodes.ainodes_backend import pil_image_to_pixmap
from custom_nodes.ainodes_engine_base_nodes.image_nodes.output_node import ImagePreviewNode
from custom_nodes.ainodes_engine_base_nodes.video_nodes.video_save_node import VideoOutputNode

from ..deforum_helpers.render import deforum_ainodes_adapter

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

    def evalImplementation_thread(self, index=0):
        data = self.getInputData(0)
        success = None
        try:
            success = deforum_ainodes_adapter(self, data)
        except:
            handle_ainodes_exception()
        return success

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
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)

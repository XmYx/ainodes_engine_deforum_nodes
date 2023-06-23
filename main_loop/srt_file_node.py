"""from qtpy import QtCore, QtWidgets
from ainodes_frontend.base import register_node, get_next_opcode, handle_ainodes_exception
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

from ainodes_frontend import singleton as gs

OP_NODE_DEFORUM_RUN = get_next_opcode()
OP_NODE_DEFORUM_CNET = get_next_opcode()

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
    op_code = OP_NODE_DEFORUM_SRT
    op_title = "Deforum Runner"
    content_label_objname = "deforum_runner_node"
    category = "aiNodes Deforum/DeForum"

    def __init__(self, scene):
        super().__init__(scene, inputs=[6,1], outputs=[6,1])

    def initInnerClasses(self):
        self.content = DeforumRunWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 300
        self.grNode.height = 250
        self.content.setMinimumWidth(300)
        self.content.setMinimumHeight(250)
        self.content.eval_signal.connect(self.evalImplementation)

    def evalImplementation_thread(self, index=0):
        return None

    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        #super().onWorkerFinished(None)
        if gs.should_run:
            if len(self.getOutputs(0)) > 0:
                self.executeChild(output_index=0)"""

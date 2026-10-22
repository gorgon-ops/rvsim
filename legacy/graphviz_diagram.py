from graphviz import Digraph

dot = Digraph("Processor", format="png")
dot.attr(rankdir="LR", size="8,5")

dot.node("IM", "Instruction\nMemory")
dot.node("CU", "Control\nUnit")
dot.node("RF", "Register\nFile")
dot.node("ALU", "ALU")
dot.node("DM", "Data\nMemory")

dot.edge("IM", "CU", label="Instruction")
dot.edge("CU", "RF", label="Control")
dot.edge("RF", "ALU", label="Operands")
dot.edge("ALU", "DM", label="Address/Data")
dot.edge("DM", "RF", label="Load Data")

dot

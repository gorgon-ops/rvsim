from manim import *

class RiscVPipeline(Scene):
    def construct(self):
        stages = ["IF", "ID", "EX", "MEM", "WB"]
        boxes = []

        for i, stage in enumerate(stages):
            box = Rectangle(width=2.2, height=1.2)
            label = Text(stage, font_size=28)
            group = VGroup(box, label).move_to(RIGHT * i * 2.5)
            boxes.append(group)

        pipeline = VGroup(*boxes).center()
        self.play(Create(pipeline))

        instr = Rectangle(width=1.6, height=0.8, color=YELLOW)
        instr_label = Text("ADD x3,x1,x2", font_size=20)
        instruction = VGroup(instr, instr_label)
        instruction.move_to(boxes[0].get_center())

        self.play(FadeIn(instruction))

        for i in range(1, len(boxes)):
            self.play(
                instruction.animate.move_to(boxes[i].get_center()),
                run_time=0.8
            )

        self.wait(1)

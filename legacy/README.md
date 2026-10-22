# Original prototypes

Kept for reference. These are the files this project grew out of:

- `pipeline_toy.py` - the first 5-stage pipeline sketch. Instructions were
  symbolic (a name plus sets of registers read and written), hazards were
  detected by comparing those sets, and nothing was ever executed. The
  cycle-by-cycle table it printed lives on as `python -m rvsim pipeline`,
  which shows the same thing for instructions that actually run.
- `graphviz_diagram.py` - a static block diagram. Superseded by the live
  datapath schematic in `web/index.html`.
- `manim_scene.py` - an animation of one instruction moving through five
  boxes. Superseded by the trace-driven visualiser, which animates real
  values from a real run.

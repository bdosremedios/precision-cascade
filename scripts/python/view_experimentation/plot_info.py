from bokeh.palettes import Category20

cat20 = Category20[20]

SOLVER_CLR_DICT = {
    "FP FP64": cat20[0],
    "FP FP32": cat20[2],
    "FP FP16": cat20[4],
    "PC HSD ORC": cat20[6],
    "PC HSD RRT": cat20[8],
    "PC HSD CS": cat20[10],
    "PC HSD S2T": cat20[12],
    "PC SD ORC": cat20[14],
    "PC SD RRT": cat20[16],
    "PC SD CS": cat20[18],
}

SOLVER_FMT_DICT = {
    "FP FP64": ",-",
    "FP FP32": ",-",
    "FP FP16": ",-",
    "PC HSD ORC": ",-",
    "PC HSD RRT": ",-",
    "PC HSD CS": ",-",
    "PC HSD S2T": ",-",
    "PC SD ORC": ",-",
    "PC SD RRT": ",-",
    "PC SD CS": ",-",
}
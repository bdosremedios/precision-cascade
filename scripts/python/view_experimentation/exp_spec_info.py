ALL_MATRICES = [
    "af_0_k101", "af_shell9", "af23560", "airfoil_2d",
    "apache2", "appu", "atmosmodj", "BenElechi1", "bone010",
    "Bump_2911", "cage10", "cage13", "cage14", "chipcool1",
    "circuit_3", "coupled", "crankseg_1", "CurlCurl_2", "CurlCurl_4",
    "e40r0100", "ecology2", "epb1", "F1", "FEM_3D_thermal1",
    "FEM_3D_thermal2", "G3_circuit", "garon2", "hood", "inlet",
    "jan99jac040sc", "language", "marine1", "mc2depi", "memplus",
    "ns3Da", "parabolic_fem", "poisson3Da", "poisson3Db", "powersim",
    "pwtk", "rajat15", "rajat31", "shermanACb", "sme3Da",
    "stomach", "t2em", "thermal2", "tmt_unsym", "torso2",
    "torso3", "TSOPF_RS_b39_c7", "venkat01", "viscoplastic2", "wang3",
    "wang4", "Zhao1", "Zhao2"
]
SMALL_MATRICES = [
    "af23560", "airfoil_2d", "appu", "cage10", "chipcool1",
    "circuit_3", "coupled", "e40r0100", "epb1", "FEM_3D_thermal1",
    "garon2", "inlet", "jan99jac040sc", "memplus", "ns3Da",
    "poisson3Da", "powersim", "rajat15", "shermanACb", "sme3Da",
    "TSOPF_RS_b39_c7", "viscoplastic2", "wang3", "wang4", "Zhao1",
    "Zhao2"
]
ILU0_MATRICES = [
    "af23560", "airfoil_2d", "appu", "cage10", "chipcool1",
    "epb1", "FEM_3D_thermal1", "inlet", "memplus", "poisson3Da",
    "powersim", "sme3Da", "viscoplastic2", "wang3", "wang4",
    "Zhao1", "Zhao2"
]

RAW_SETUPS = ["ilu0", "ilutp_1em2", "ilutp_1em4", "unpreconditioned_dense", "unpreconditioned"]
RAW_SOLVERS = [
    "FP16",
    "FP32",
    "FP64",
    "SD_OuterRestartCount",
    "SD_CheckStagnation",
    "SD_RelativeResidualThreshold",
    "OuterRestartCount",
    "CheckStagnation",
    "RelativeResidualThreshold",
    "StagnationToThreshold"
]

SETUP_TO_ID_MAPPING = {
    "ilu0": "ilu0",
    "ilutp_1em2": "ilutp1em2",
    "ilutp_1em4": "ilutp1em4",
    "unpreconditioned_dense": "unpreconddense",
    "unpreconditioned": "unprecond"
}
SETUP_MATRIX_MAPPING = {
    "ilu0": ILU0_MATRICES,
    "ilutp1em2": SMALL_MATRICES,
    "ilutp1em4": SMALL_MATRICES,
    "unpreconddense": SMALL_MATRICES,
    "unprecond": ALL_MATRICES
}
SOLVER_TO_ID_MAPPING = {
    "FP16": "FP FP16",
    "FP32": "FP FP32",
    "FP64": "FP FP64",
    "SD_OuterRestartCount": "PC SD ORC",
    "SD_CheckStagnation": "PC SD CS",
    "SD_RelativeResidualThreshold": "PC SD RRT",
    "OuterRestartCount": "PC HSD ORC",
    "CheckStagnation": "PC HSD CS",
    "RelativeResidualThreshold": "PC HSD RRT",
    "StagnationToThreshold": "PC HSD S2T"
}

N_EXPERIMENT_ITERATIONS = 3
REL_RES_TOL = 1e-12
RESTART_PARAMS = [10, 20, 30, 40, 50, 100, 150, 200]
SETUPS = ["unprecond", "unpreconddense", "ilu0", "ilutp1em2", "ilutp1em4"]
FP_SOLVERS = ["FP FP16", "FP FP32", "FP FP64"]
PC_SOLVERS = [
    "PC SD ORC", "PC SD CS", "PC SD RRT",
    "PC HSD ORC", "PC HSD CS", "PC HSD RRT", "PC HSD S2T"
]
GMRES_M_SOLVERS = ["FP FP64"] + PC_SOLVERS
SOLVERS = FP_SOLVERS + PC_SOLVERS
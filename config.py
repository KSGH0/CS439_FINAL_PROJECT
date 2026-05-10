SEED                     = 2_026
ALPHA                    = 1.0    # Laplace smoothing (standard)
VOCAB_SIZE               = 10_000  # max features for TF-IDF; total observed corpus vocabulary is 266,760
LR_RATE                  = 0.01   # stable for high-dim TF-IDF
N_ITER                   = 500    # more iterations for better convergence
LAMBDA                   = 10.0   # stronger L2 regularization
TRAIN_SIZE               = 0        # 0 = all 288K papers for NLP; >0 = specific count
COST_FRACTION            = 0.25
RESOURCE_COST            = 500     # flat compute/equipment cost per paper
SALARY_MULTIPLIER        = 1.0     # global salary scaler: 1.0=published medians, 0.7=30% lower, 1.3=30% higher

# Salary tiers (US dollars, annual). All scaled by SALARY_MULTIPLIER at runtime.
SALARY_PROFESSOR         = 96_430    # professor / PI — median
SALARY_GRAD              = 48_500    # graduate researcher (PhD + postdoc combined, average of both medians)
SALARY_MASTERS           = 20_000    # master student — unknown, set below PhD level; adjust as needed
SALARY_UNDERGRAD         = 0         # undergrad — typically unpaid / course credit

# Authorship distribution for middle-author weighted average. Must sum to 1.0.
# Co-dependent with the SALARY_* values above: changing one without the other
# breaks the cost model. Edit both together when tuning.
DIST_PROFESSOR           = 0.15   # 15% professors / PIs
DIST_GRAD                = 0.7   # 70% graduate researchers (PhD + postdoc combined)
DIST_MASTERS             = 0.05   # 5%  master students
DIST_OTHER               = 0.05   # 5%  visiting researchers, RAs, other unknown
DIST_UNDERGRAD           = 0.05   # 5%  undergrads
SALARY_OTHER_ACADEMIC    = 48_500  # salary for the "other/unknown" bucket
MONTHS_PER_PAPER         = 6
DATA_PATH                = "archive/"
OUTPUT_DIR               = "outputs/"   # final deliverables: PNGs, model_results.csv
INPUT_DIR                = "inputs/"    # intermediate data flowing between scripts

FILES = [
    "cs_ai_papers.jsonl",
    "cs_lg_papers.jsonl",
    "cs_cl_papers.jsonl",
    "cs_ne_papers.jsonl",
    "cs_ir_papers.jsonl",
    "cs_cv_papers.jsonl",
]

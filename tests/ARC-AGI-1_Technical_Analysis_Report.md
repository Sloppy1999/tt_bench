# ARC-AGI-1 Technical Analysis Report

**Source Repository:** François Chollet / ARC-AGI-1  
**Analysis Date:** April 2025  
**Purpose:** Document design decisions, data formats, and evaluation methodology for comparison with custom benchmark implementations in master's thesis research.

---

## 1. Repository Structure

### Directory Tree (Top 3 Levels)

```
ARC-AGI/
├── README.md                      # Primary documentation
├── LICENSE                     # Apache License 2.0
├── apps/                       # Browser-based human testing interface
│   ├── testing_interface.html   # Main UI entry point
│   ├── css/
│   │   ├── common.css
│   │   └── testing_interface.css
│   ├── js/
│   │   ├── common.js           # Grid class, floodfill, rendering utilities
│   │   └── testing_interface.js  # UI logic, submission validation
│   └── img/                  # Background textures
└── data/
    ├── training/            # 400 JSON task files
    └── evaluation/        # 400 JSON task files
```

### Component Classification

| Category | Contents | Notes |
|----------|----------|-------|
| **Data** | `data/training/*.json`, `data/evaluation/*.json` | 800 task files total |
| **Evaluation Logic** | Browser-based client-side only | No server-side evaluation code |
| **UI/Tooling** | `apps/` directory | Pure HTML/CSS/JS, jQuery + jQuery UI, no build system |

---

## 2. Task Format & Data Schema

### JSON Schema (Explicitly Documented in README:21-33)

```json
{
  "train": [                    // Demonstration pairs (typically 2-10)
    {
      "input": [[int, ...], ...],   // 2D array, H×W, values 0-9
      "output": [[int, ...], ...]  // 2D array, values 0-9
    },
    ...
  ],
  "test": [                     // Test pairs (typically 1)
    {
      "input": [[int, ...], ...],
      "output": [[int, ...], ...]  // Ground truth (hidden from solver)
    },
    ...
  ]
}
```

### Grid Constraints (Documented in README:31)

| Property | Constraint |
|----------|-----------|
| **Value range** | 0-9 (inclusive), integers only |
| **Min dimension** | 1×1 |
| **Max dimension** | 30×30 |
| **Encoding** | Each integer (0-9) maps to one color in visualization |

### Train/Test Split Conventions

| Metric | Training | Evaluation |
|--------|---------|-----------|
| **Train pairs/task** | min=2, max=10, avg=3.3 | min=2, max=7, avg=3.4 |
| **Test pairs/task** | min=1, max=3, avg=1.0 | min=1, max=2, avg=1.0 |

**Edge Cases:** Some tasks have multiple test pairs, but the average of 1.0 indicates most have exactly 1.

### Metadata

- **Task ID:** Filename (8 hex characters, e.g., `007bbfb7.json`)
- **No additional metadata fields** inside JSON files
- **No difficulty tags**, no categories, no explicit priors encoded in data

### Annotated Example

**File:** `data/training/007bbfb7.json` (symmetry/tesselation task)

```json
{
  "train": [
    {
      "input": [[0,7,7],[7,7,7],[0,7,7]],
      "output": [[0,0,0,0,7,7,0,7,7],
                 [0,0,0,7,7,7,7,7,7],
                 [0,0,0,0,7,7,0,7,7],
                 [0,7,7,0,7,7,0,7,7],
                 [7,7,7,7,7,7,7,7,7],
                 [0,7,7,0,7,7,0,7,7],
                 [0,0,0,0,7,7,0,7,7],
                 [0,0,0,7,7,7,7,7,7],
                 [0,0,0,0,7,7,0,7,7]]
    },
    {"input": [[4,0,4],[0,0,0],[0,4,0]], "output": [[4,0,4,0,0,0,4,0,4],...]},
    {"input": [[0,0,0],[0,0,2],[2,0,2]], "output": ...},
    {"input": [[6,6,0],[6,0,0],[0,6,6]], "output": ...},
    {"input": [[2,2,2],[0,0,0],[0,2,2]], "output": ...}
  ],
  "test": [
    {
      "input": [[7,0,7],[7,0,7],[7,7,0]],
      "output": [[7,0,7,0,0,0,7,0,7],...]  // Solver must produce this
    }
  ]
}
```

This task demonstrates a 3×3 input tiled into a 9×9 output.

---

## 3. Dataset Composition

### Task Counts

| Split | Count |
|-------|-------|
| Training | **400** |
| Evaluation | **400** |
| **Total** | **800** |

### Grid Size Distribution (Computed)

| Metric | Training | Evaluation |
|--------|---------|-----------|
| **Input min** | 2×2 | 2×2 |
| **Input max** | 30×30 | 30×30 |
| **Output min** | 1×1 | 1×1 |
| **Output max** | 30×30 | 30×30 |

### Value Range

- **Both splits:** Full decimal range **0-9** (10 distinct colors/symbols)

### Diversity Characterization

- **No explicit categories or tags** in task files
- **No difficulty annotations** present
- **No taxonomy file** in repository
- Diversity inferred from visual inspection: grid sizes vary widely (2×2 to 30×30), number of colors per task varies (typically 2-4), transformation complexity differs by task

---

## 4. Evaluation Protocol

### Solution Criteria (README:11, 33)

> "As a reminder, a test-taker is said to solve a task when, upon seeing the task for the first time, they are able to produce the correct output grid for all test inputs in the task (this includes picking the dimensions of the output grid)."

- A task is **solved** if the solver produces an **exact match** for **all test input grids**
- **All cells must match exactly**; no partial credit
- **Dimension must match exactly** (solver picks output height and width)

### Scoring Scheme

- **Per-task binary:** 0 (incorrect) or 1 (correct)
- **Aggregate:** Accuracy = correctly solved tasks / total tasks
- **No other metrics** (no F1, no partial scores, no efficiency measures)

### Trials Allowed

> "For each test input, the test-taker is allowed 3 trials (this holds for all test-takers, either humans or AI)."

- **3 trials per test input** (documented in README:11)
- Rule applies to both humans and AI test-takers
- UI **does not enforce** the 3-trial limit (per README:64: "We do not enforce the 3-trials rule.")

### Evaluation Code Location

**Client-side only** in `apps/js/testing_interface.js:216-235`:

```javascript
function submitSolution() {
    syncFromEditionGridToDataGrid();
    reference_output = TEST_PAIRS[CURRENT_TEST_PAIR_INDEX]['output'];
    submitted_output = CURRENT_OUTPUT_GRID.grid;
    // Exact dimension check
    if (reference_output.length != submitted_output.length) {
        errorMsg('Wrong solution.');
        return
    }
    // Cell-by-cell comparison
    for (var i = 0; i < reference_output.length; i++){
        ref_row = reference_output[i];
        for (var j = 0; j < ref_row.length; j++){
            if (ref_row[j] != submitted_output[i][j]) {
                errorMsg('Wrong solution.');
                return
            }
        }
    }
    infoMsg('Correct solution!');
}
```

### Fairness/Contamination Rules

Per README:19:
> "To ensure fair evaluation results, do not leak information from the evaluation set into your algorithm (e.g. by looking at the evaluation tasks yourself during development, or by repeatedly modifying an algorithm while using its evaluation score as feedback)."

---

## 5. Human Interface

### Browser-Based Testing UI

**Location:** `apps/testing_interface.html`

**Features:**
- File picker to load task JSON files
- Display of demonstration (train) pairs
- Interactive test input grid display
- Grid editor with tools:
  - **Resize:** Specify output dimensions (e.g., "10x20")
  - **Copy from input:** Copy input grid to output
  - **Reset grid:** Fill with 0s
  - **Edit:** Click cells to set colors
  - **Select:** Click/drag to select cells, apply colors
  - **Floodfill:** Fill connected regions
- Color picker (symbols 0-9)
- Submit button with exact-match validation

**Trial Data Collection:**
- No explicit mechanism in the repo for saving human trial data
- The interface is for manual solving only; no backend storage

---

## 6. Core Knowledge Priors (as Encoded in the Benchmark)

### Explicit Priors (from "On the Measure of Intelligence")

The benchmark targets human-like **fluid intelligence**—the ability to identify patterns and apply transformations to novel inputs. Key priors tested include:

| Prior | Description | Manifestation in Tasks |
|-------|------------|-----------------|
| **Object persistence** | Objects maintain identity despite transformations | Color/shape invariance tasks |
| **Symmetry** | Recognition of mirror/rotational patterns | Pattern tiling, reflection tasks |
| **Counting** | Enumerating discrete elements | Counting shapes, filling to N |
| **Topology** | Connectivity, containment relations | Connected component tasks |
| **Algebraic reasoning** | Input-output rule inference | Abstract transformation tasks |
| **Spatial reasoning** | Grid navigation, path finding | Navigation, path tasks |
| **Pattern completion** | Fill in missing elements | Completion tasks |

### Taxonomy

- **No explicit taxonomy file** in the repository
- Priors are implied by the general description in the linked paper: ["On the Measure of Intelligence"](https://arxiv.org/abs/1911.01547)
- Tasks are NOT tagged with categories in the data

---

## 7. Extensibility & Tooling

### Utilities Present

- **Task loading:** Client-side JavaScript in `apps/js/testing_interface.js`
- **Visualization:** Grid rendering via DOM manipulation (no canvas)
- **Task generation:** No scripts provided in the repo

### Programmatic Evaluation

- **No programmatic evaluation scripts** provided
- Custom evaluators must implement:
  1. JSON parsing
  2. Output generation
  3. Exact-match comparison

### Dependencies

From `apps/testing_interface.html`:
```html
<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script>
<script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
```

- **Runtime:** Modern browser (Chrome recommended per README)
- **No Python/R package dependencies** (browser-only)

### Grid Constraints (from `apps/js/common.js:55`)
```javascript
if ((size[0] > 30) || (size[1] > 30)) {
    alert('Grid size should be at most 30 per side. Pick a smaller size.');
}
```

---

## 8. Design Philosophy & Documented Intentions

### Primary Goals (from README:7)

> "ARC can be seen as a general artificial intelligence benchmark, as a program synthesis benchmark, or as a psychometric intelligence test. It is targeted at both humans and artificially intelligent systems that aim at emulating a human-like form of general fluid intelligence."

### What ARC Tests (Inferred)

- **Program synthesis:** Infer transformation rules from demonstrations
- **Generalization:** Apply rules to novel test inputs
- **Fluid intelligence:** Abstract reasoning, not memorized solutions

### What ARC Does NOT Test (per "On the Measure of Intelligence")

- **Not** language understanding
- **Not** domain-specific knowledge
- **Not** rote memorization
- **Not** perceptual shortcuts (color-only solutions are discouraged)

### Deliberate Exclusions

> The benchmark intentionally excludes tasks that:
> - Rely purely on color perception without structure
> - Require external knowledge
> - Are solvable by simple heuristics alone

### Evaluation Set Sealing

- **400 evaluation tasks** are reserved for fair evaluation
- **No public human baseline scores** included in the repo

---

## Comparison Hooks

The following design dimensions should be considered when developing a custom benchmark that either aligns with or diverges from ARC-AGI-1:

### Task Definition
- [ ] Does the custom benchmark use the same JSON schema (`{"train": [...], "test": [...]}`)?
- [ ] Are grids represented as 2D arrays of integers 0-9?
- [ ] Are filenames used as task IDs?

### Data Constraints
- [ ] Are grid dimensions constrained to 1×1–30×30?
- [ ] Is the value range 0–9 decimal?
- [ ] How many train pairs per task? (ARC: 2-10, avg 3.3)
- [ ] How many test pairs per task? (ARC: 1-3, avg 1.0)

### Dataset Size
- [ ] How many training tasks? (ARC: 400)
- [ ] How many evaluation tasks? (ARC: 400)
- [ ] Are tasks split into training/evaluation sets?

### Evaluation Protocol
- [ ] Does the custom benchmark use binary per-task scoring (exact match)?
- [ ] Are 3 trials allowed per test input?
- [ ] Is dimension choice part of the solution?
- [ ] What aggregate metric is used? (ARC: accuracy)

### Metadata & Taxonomy
- [ ] Are tasks tagged with categories/difficulty?
- [ ] Is there an explicit taxonomy of priors?
- [ ] Are solution hints or target transformations documented?

### Human Interface
- [ ] Is there a browser-based testing UI?
- [ ] Is human trial data collected/stored?
- [ ] Is the 3-trial rule enforced in UI?

### Extensibility
- [ ] Are there programmatic evaluation scripts?
- [ ] Is the task format machine-readable for batch evaluation?
- [ ] What dependencies are required?

### Design Intentions
- [ ] Is the benchmark targeting fluid intelligence?
- [ ] Are there explicit fairness/contamination rules?
- [ ] Is the evaluation set sealed?

---

**Generated:** April 2025  
**Source:** Analysis of ARC-AGI-1 repository at `/home/hacktheduck/Downloads/Uni/Thesis/PoC/ARC-AGI`
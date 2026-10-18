# Official guide corpus

Ground truth: `Papers/practice-guide-2021.pdf`, Practice Guide 1.0.
`INDEX.json` lists 55 playable boards and two paper exercises. PDF page numbers
in provenance are 1-based; the printed page number is four less.

## Layout and inventories

`challenges/json/` contains the playable boards. `exercises/` contains references
to the binary-register worksheets (21-A and 22-A), which have no playable board.
The unsupported 1-A entry was removed from the official board set; its original
is preserved under `review/official_pdf_review/repair/before_install/`.

`board.fixed_components` describes the starting diagram. Other solution parts
are in `solution.placed_components`. Hopper counts come from the starting-setup
panel, not the examples or the illustrated output strip.

`available_parts.ramp` is one shared physical pool. A `ramp_left` or `ramp_right`
placement each consumes one ramp. Other keys are individual component pools.
`source_inventory` records the printed quantities, including `"unlimited"`.
An unlimited ramp supply has an effective limit of 121 in `available_parts`,
since an 11×11 board cannot hold more than 121 components. A start-button icon
specifies the initial release, not an on-grid `trigger` part.

## State-selection puzzles

Where the starting diagram leaves a bit's direction open for the solver,
`board.editable_bit_states` lists its fixed coordinates. Its stored state is
only the display/default state. The solution must explicitly assign every such
bit through an entry with the same type and coordinates in
`solution.placed_components`. These entries set initial states; they consume no
inventory and do not permit moving or removing the fixed bits.

The agent can make those assignments with `place_component`. Trials that vary
starting conditions (logic and reversal puzzles) instead use
`trials[].initial_bit_states`; those states are controlled by the evaluator.

## Targets

- `required_output` records physical marble colours, with `intercepted` for a
  terminal interception. This is distinct from the historical catcher-based
  `solution.final_marble_state`. When both are supplied, both are checked.
- `trials` express multiple starting conditions or hopper counts. Every trial
  must follow a legal path and satisfy its expectations.
- `expect.intercepted_at` names an interceptor's coordinates;
  `expect.intercepted_colours` names the intercepted marble colours.
- Register bit lists are ordered most-significant first. The guide's vertical
  counters have the least-significant bit at the top.
- Logic puzzles cover all four inputs; reversal puzzles cover every initial
  bit configuration; counting/subtraction cover 0–15; overflow covers 0–32,
  including the printed examples and repeated overflow boundaries.

`solution.verified` records simulator verification of the reference solution.
It is not a certification of companion question answers or derived datasets.
The repair record is `review/official_pdf_review/repair/RESULTS.md`.

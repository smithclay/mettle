# General Grouped Aggregate Engine

This document defines the grouped aggregate engine we want in `mettle`.

It is not a Q3 design. Q3 remains a benchmark and one admitted fast path, but the grouped aggregate substrate must be generic across tables, schemas, and eventually datasets.

## Architecture Statement

The grouped aggregate engine has one semantic surface:

`GroupedAggregatePlan = group keys + aggregate slots + output layout`

Everything below that line is a physical strategy choice.

That means:

- SQL shape decides grouping semantics once
- output aliases and ordering are owned by the grouped aggregate plan
- CPU hash aggregation, GPU hash aggregation, and later sort aggregation are interchangeable physical implementations
- representative-row materialization and top-k are downstream consumers, not the grouping semantics

## Semantic Contract

The grouped aggregate engine must answer:

- which rows belong to the same group
- which aggregate states are maintained per group
- how final result columns are assembled

The semantic plan is:

```rust
struct GroupedAggregatePlan {
    group_by: Vec<String>,
    aggregate_slots: Vec<AggregateSlotPlan>,
    outputs: Vec<GroupedAggregateOutput>,
}

struct AggregateSlotPlan {
    alias: String,
    func: AggFunc,
    input: AggregateInputPlan,
    state_kind: AggregateStateKind,
}

enum GroupedAggregateOutput {
    GroupKey { group_idx: usize, alias: String },
    Aggregate { agg_idx: usize },
}
```

This contract is intentionally generic:

- output order may interleave group keys and aggregates
- aggregate state is independent of output order
- `COUNT(*)` is a first-class input form
- `AVG` is represented as a compound state (`sum`, `count`) even if the final output is one scalar

## Physical Model

The physical engine should be decomposed into three layers.

### 1. Key Layout

Convert each row into a comparable group key.

Required properties:

- exact equality semantics for admitted key types
- support for single-key and composite-key grouping
- dictionary codes are valid key lanes for string dimensions
- hash is for addressing only; key equality decides group identity

The right abstraction is a `GroupKeyLayout`, not a benchmark-specific tuple.

### 2. Aggregate State Layout

Each aggregate slot maps to one or more state lanes:

- `COUNT` -> `u64 count`
- `SUM` -> `f64 sum`
- `MIN` -> typed running minimum
- `MAX` -> typed running maximum
- `AVG` -> `{ f64 sum, u64 count }`

This should be modeled as an `AggregateStateLayout` derived from `aggregate_slots`.

The engine updates state per group, then finalizes state into output columns.

### 3. Output Assembly

Final result assembly uses `outputs`, not raw SQL projection scanning.

That is the boundary that makes the engine generic:

- group key columns are gathered from representative rows or decoded from stored key lanes
- aggregate outputs are finalized from state
- order and aliases come from `GroupedAggregatePlan`

## Physical Strategies

The engine should support multiple strategies under the same plan.

### Strategy A: CPU Hash Aggregate

This is the correctness baseline.

- exact composite key map
- all current aggregate functions
- any admitted grouped aggregate query can execute here

### Strategy B: GPU Hash Aggregate

This is a fast path, not the definition of grouping.

Near-term scope:

- numeric or dictionary-coded key lanes
- exact key comparison
- specialized kernels for common state layouts
- fallback when a layout is unsupported

Today `SUM` on numeric keys is one such specialization. It should stay a specialization, not become the public abstraction.

### Strategy C: Streaming Or Sort Aggregate

Later, the same semantic plan can be lowered to a sort-based or partitioned aggregate when that becomes cheaper than a large hash table.

That choice belongs in physical planning, not in SQL parsing.

## Planning Rule

Grouped aggregation should be planned in two steps:

1. Build `GroupedAggregatePlan` from the parsed SQL plan.
2. Choose a physical aggregate strategy based on key layout, aggregate state layout, and available operators.

This is the key separation that keeps the engine generic.

## Current Direction In Code

The codebase should use `GroupedAggregatePlan` as the canonical grouped aggregate description.

That implies:

- grouped output schema derives from the grouped plan
- CPU grouped fallback executes from aggregate slots and outputs
- GPU grouped execution admits only the layouts it actually supports
- benchmark-specific grouped top-k code consumes grouped results instead of defining them

Current implemented GPU strategy:

- exact single-key grouping for `Int32`, `Date32`, and dictionary-coded string keys
- `SUM(expr)` over device-evaluated `GpuVector` inputs
- fallback for composite keys and non-SUM state layouts

That is intentionally narrower than the semantic engine. The point is to specialize by physical layout, not by benchmark query.

## Non-Goals

This design does not require:

- one kernel that handles every aggregate combination
- one GPU path that works for every dataset immediately
- removing fallback paths

It does require one generic semantic engine that can add physical strategies without rewriting grouped aggregate semantics.

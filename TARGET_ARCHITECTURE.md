# Target GPU Architecture

This document is the architecture for the next `mettle` fast path. It is intentionally narrower than a full GPU SQL engine.

The target is specific: beat DuckDB on pinned TPC-H Q3 at SF=1 without turning the codebase into a second system.

## Architecture Statement

The hot path has one governing rule:

`A supported query remains a DeviceRelation from the first filter until the final result materialization.`

That means:

- the host may parse, plan, and choose a fallback
- the host may download final output rows
- the host may not inspect or resize intermediate relations in the happy path
- the host may not rebuild post-join state in a CPU `HashMap`

Everything else in this document serves that rule.

## Success Criteria

- Warm pinned Q3 median under 6.0 ms on the same machine and harness.
- No host `download_count()` between joins.
- No CPU aggregate in the Q3 happy path.
- No CPU string filter in the Q3 happy path.
- Final host download proportional to result size, not post-join cardinality.
- Existing fallback behavior preserved for unsupported queries.

## Scope

This architecture applies only to a narrow pilot path:

- pinned tables only
- inner equi-joins only
- normalized scalar columns only
- non-null filter, join, and group-by columns only
- dictionary equality on pinned string columns
- arithmetic expressions that can already be represented in the current GPU expression path
- grouped aggregate needed for Q3
- `ORDER BY ... LIMIT K` handled as top-k plus final exact ordering over a small candidate set

Anything outside this scope falls back to the current execution path. That is a design rule, not a temporary apology.

## Why The Current Path Tops Out

The current implementation already has useful GPU operators, but the control flow is still host-shaped in the hottest path:

- joins stop so the host can read `download_count()` and rewrite `max_len`
- grouped aggregate downloads post-join columns and builds a CPU `HashMap`
- string equality filters are computed on CPU and uploaded back to GPU
- SQL source order still effectively determines join order

Those are the costs this architecture removes. It does not attempt to solve every GPU database problem at once.

## Core Abstractions

### `PinnedTable`

`PinnedTable` is the immutable catalog entry for a pinned DuckDB table.

Its contract is:

- normalized column storage
- optional long-lived device buffers
- dictionary encoding for pinned string columns
- lightweight stats used only for physical planning
- explicit metadata about whether a column is admissible to the GPU pilot path

Required metadata:

```rust
struct TableStats {
    row_count: usize,
    columns: HashMap<String, ColumnStats>,
}

enum ColumnStats {
    Numeric { min: i64, max: i64, distinct_estimate: u64 },
    Float { min: f32, max: f32, distinct_estimate: u64 },
    Dictionary { distinct_count: u32 },
    Unknown,
}
```

Pilot-path admission rule:

- if a filter, join, or group-by column is nullable, unsupported, or not pinned, the query falls back

That rule is deliberate. The current pinned representation does not model null semantics cleanly enough to promise them on GPU.

### `DeviceRelation`

`DeviceRelation` is the only intermediate value allowed in the fast path.

```rust
struct DeviceRelation {
    sources: Vec<SourceBinding>,
    selections: Vec<SelectionVector>,
    card: DeviceCardinality,
}
```

Its invariants are the center of the architecture:

- `card.count` is the only authoritative row count
- all `selections` represent exactly `card.count` rows
- cardinality changes are written by device operators, not by host rewrites
- no column values are downloaded while a relation is still being transformed

If an operator cannot preserve those invariants, that query shape is not in the pilot path.

### `SelectionVector`

`SelectionVector` maps a source table into the current `DeviceRelation`.

```rust
struct SelectionVector {
    indices: GpuBuffer,
    card: DeviceCardinality,
}
```

Contract:

- `indices[i]` is a row id in the original source table
- `card` matches the enclosing relation cardinality
- reindexing composes selections on device only

### `DeviceCardinality`

`DeviceCardinality` exists to eliminate host-managed launch bounds.

```rust
struct DeviceCardinality {
    count: GpuBuffer,
    dispatch_args: GpuBuffer,
    capacity: usize,
}
```

Contract:

- `count` is written on device
- `dispatch_args` is written on device for the next operator launch
- the host may read `count` only after the final materialization boundary

This is the direct architectural replacement for host-managed `max_len`.

### `PhysicalGpuPlan`

`PhysicalGpuPlan` is a short sequence of operators over `DeviceRelation`.

```rust
enum PhysicalOp {
    Filter(FilterOp),
    Join(JoinOp),
    Project(ProjectOp),
    Aggregate(AggregateOp),
    TopK(TopKOp),
    Materialize(MaterializeOp),
}
```

This is not an extensibility surface. For the pilot path, these are the only operators that matter.

## Operator Contracts

### Filter

Input:

- one `DeviceRelation`
- one filterable column from a pinned source

Output:

- one `DeviceRelation`

Rules:

- numeric and date predicates run as scalar compares
- string equality runs as dictionary-code equality
- multiple predicates intersect selections on device
- any unsupported predicate falls back before execution starts

### Join

Input:

- two relations connected by an inner equi-join

Output:

- one `DeviceRelation`

Pilot-path semantics:

- build on the smaller filtered side
- use the GPU path only when the build key is known unique for the admitted pilot workload
- if duplicate-key general join semantics are required, fall back

This is intentionally narrow. The current system should first prove that the architecture wins on the PK/FK-heavy workload it is targeting.

### Project

Input:

- one `DeviceRelation`

Output:

- one `DeviceRelation` with projected value buffers needed by downstream aggregate or materialize

Rules:

- projection never forces host materialization
- projection may gather source columns through selections
- expressions remain limited to the arithmetic already supported by the current engine

### Aggregate

Input:

- one `DeviceRelation`

Output:

- one grouped `DeviceRelation`

Pilot-path semantics:

- exact group semantics, not hash-only semantics
- Q3 group key is represented as a fixed-width 3-lane key
- aggregate state remains on device
- `SUM` is required for the pilot path
- `COUNT`, `MIN`, `MAX`, and `AVG` are deferred until the pilot path is proven

Required entry shape for the Q3 pilot:

```rust
struct GroupEntry {
    hash: u64,
    key0: u32,
    key1: u32,
    key2: u32,
    revenue_sum: f32,
    occupied: u32,
}
```

The hash is for addressing only. Full key comparison determines equality.

### Top-K And Materialize

Input:

- one grouped `DeviceRelation`

Output:

- final host rows

Rules:

- `TopK` reduces candidate rows on device
- final exact ordering may be finished on host over a small candidate set
- host download must remain `O(K)` for `LIMIT K`

That last rule matters more than whether the final tie-break happens on GPU or CPU.

## Planning Rules

The planner is intentionally simple.

### Admission

A query enters the pilot path only if all of these are true:

- all referenced tables are pinned
- all required columns satisfy pilot-path type and nullability rules
- the query shape is expressible as `Filter -> Join -> Project -> Aggregate -> TopK -> Materialize`
- join keys satisfy the pilot join contract, including known-unique build-side semantics

Otherwise the query falls back immediately.

### Physical Choices

For admitted queries, the planner may choose only:

- join order
- build side
- whether top-k is required
- whether the final exact ordering can be completed over the reduced candidate set

The planner does not attempt broad cost-based optimization. It is a small physical chooser for one workload family.

### Estimation

Use only lightweight stats:

- row count
- min/max for numeric and date filters
- dictionary cardinality for string equality
- distinct estimate for join keys

That is enough for a greedy filtered-cardinality join order. More is deferred.

## Q3 Pilot Path

The first end-to-end fast path is exactly this:

1. Filter `customer` on dictionary-coded `c_mktsegment = 'HOUSEHOLD'`.
2. Filter `orders` on `o_orderdate < date '1995-03-25'`.
3. Join filtered `customer` to filtered `orders`.
4. Filter `lineitem` on `l_shipdate > date '1995-03-25'`.
5. Join the prior relation to filtered `lineitem`.
6. Project `revenue = l_extendedprice * (1 - l_discount)`.
7. Aggregate by `(l_orderkey, o_orderdate, o_shippriority)` with `SUM(revenue)`.
8. Apply top-k for `ORDER BY revenue DESC, o_orderdate ASC LIMIT 10`.
9. Download only the small final candidate set and emit the final rows.

If this path does not beat DuckDB, the architecture is not yet validated and should not be broadened.

## Explicit Non-Goals For The Pilot

The following are not part of the architecture for the first win:

- general duplicate-key GPU joins
- variable-width composite group keys
- broad aggregate coverage beyond what Q3 needs
- GPU-native scan and decode from DuckDB storage
- multi-queue scheduling
- broad kernel framework refactors
- speculative module splits not required for the pilot

These may be good ideas later. They are not part of the first architecture because they are not required to validate it.

## Implementation Sequence

### Stage 1: Catalog Admission

- extend `PinnedTable` with dictionary encoding and lightweight stats
- mark pilot-path admissible columns explicitly
- make unsupported nullability and types fall back early

Exit:

- Q3 string filter is admitted to GPU planning
- fallback boundaries are explicit and testable

### Stage 2: DeviceRelation Hot Path

- introduce `DeviceRelation`, `SelectionVector`, and `DeviceCardinality`
- make filters and joins preserve relation invariants
- remove host `download_count()` from the join loop

Exit:

- no host cardinality read between joins
- join dispatch follows device-written cardinality

### Stage 3: Exact Q3 Aggregate

- add fixed-width exact composite aggregate for the Q3 key
- keep grouped state on device

Exit:

- no CPU `HashMap` in the Q3 fast path
- no download of the 100k-row post-join intermediate

### Stage 4: Top-K Materialization

- reduce grouped output to a small candidate set on device
- finish exact final ordering over the reduced set

Exit:

- final host transfer is proportional to `LIMIT`
- Q3 benchmark becomes the only performance gate for this architecture

## Expected Payoff

The expected win comes from only three changes:

- remove join over-dispatch driven by stale host launch bounds
- remove CPU aggregate over the post-join intermediate
- remove large post-aggregate download and full host sort

If those three do not clear the benchmark target, the correct response is to measure again, not to add more architectural surface area.

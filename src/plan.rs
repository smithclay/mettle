//! SQL → GPU plan builder and executor.
//!
//! Parses user SQL via `sqlparser` into a flat `GpuPlan` pipeline IR
//! (scan → filter → join → project → aggregate → sort → limit), then
//! executes it against GPU-resident pinned tables on Metal with late
//! materialization (index arrays through joins, data gathered at output).
//! Returns `None` from `try_build_gpu_plan` for unsupported query shapes,
//! allowing fallthrough to DuckDB CPU execution.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use duckdb::arrow::array::{
    ArrayRef, Date32Array, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
};
use duckdb::arrow::datatypes::{DataType, Field, Schema};
use duckdb::arrow::record_batch::RecordBatch;

use sqlparser::ast::{
    self, BinaryOperator, Expr, FunctionArg, FunctionArgExpr, GroupByExpr, ObjectName, OrderByKind,
    SelectItem, SetExpr, Statement, UnaryOperator, Value,
};
use sqlparser::dialect::DuckDbDialect;
use sqlparser::parser::Parser;

use crate::config::BufferConfig;
use crate::host_plan;
use crate::host_plan::HostPlanHints;
use crate::metal::{
    self, GpuGroupedAggregateSpec, GpuGroupedAggregateStateKind, GpuVector, ScalarCompareOp,
    ScalarProjectOp,
};
use crate::pinned::{self, PinnedColumn, PinnedTable};
use crate::processing::MaterializedQueryResult;
use crate::source_cache;
use crate::state;

/// Flat pipeline IR for single-table and multi-table GPU queries.
#[derive(Debug, Clone)]
pub struct GpuPlan {
    pub sources: Vec<TableSource>,
    pub joins: Vec<GpuJoin>,
    pub filters: Vec<GpuFilter>,
    pub projections: Vec<GpuProjection>,
    pub aggregate: Option<GpuAggregate>,
    pub order_by: Vec<GpuOrderBy>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct TableSource {
    pub name: String,
    pub alias: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GpuJoin {
    pub left_table: String,
    pub left_col: String,
    pub right_table: String,
    pub right_col: String,
}

#[derive(Debug, Clone)]
pub struct QualifiedColumn {
    pub table: Option<String>,
    pub column: String,
}

#[derive(Debug, Clone)]
pub struct GpuFilter {
    pub table: Option<String>,
    pub column: String,
    pub op: ScalarCompareOp,
    pub literal: GpuLiteral,
}

#[derive(Debug, Clone)]
pub enum GpuLiteral {
    Int(i64),
    Float(f64),
    Str(String),
    Date(i32),
}

#[derive(Debug, Clone)]
pub struct GpuProjection {
    pub alias: String,
    pub expr: GpuExpr,
}

#[derive(Debug, Clone)]
pub enum GpuExpr {
    Column(String),
    Literal(GpuLiteral),
    BinaryOp {
        op: GpuBinOp,
        left: Box<GpuExpr>,
        right: Box<GpuExpr>,
    },
    AggCall {
        func: AggFunc,
        arg: Box<GpuExpr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBinOp {
    Add,
    Sub,
    Mul,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    Sum,
    Count,
    Min,
    Max,
    Avg,
}

#[derive(Debug, Clone)]
pub struct GpuAggregate {
    pub group_by: Vec<String>,
    /// Parsed aggregate expressions. Exercised by planner unit tests;
    /// the executor derives aggregates via a different path.
    #[allow(dead_code)]
    pub agg_exprs: Vec<GpuAggExpr>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // fields consumed by planner unit tests only
pub struct GpuAggExpr {
    pub func: AggFunc,
    pub expr: GpuExpr,
    pub alias: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AggregateStateKind {
    CountU64,
    SumF64,
    MinF64,
    MaxF64,
    AvgSumCount,
}

#[derive(Debug, Clone)]
enum AggregateInputPlan {
    CountStar,
    Expr(GpuExpr),
}

#[derive(Debug, Clone)]
struct AggregateSlotPlan {
    alias: String,
    func: AggFunc,
    input: AggregateInputPlan,
    state_kind: AggregateStateKind,
}

#[derive(Debug, Clone)]
enum GroupedAggregateOutput {
    GroupKey { group_idx: usize, alias: String },
    Aggregate { agg_idx: usize },
}

#[derive(Debug, Clone)]
struct GroupedAggregatePlan {
    group_by: Vec<String>,
    aggregate_slots: Vec<AggregateSlotPlan>,
    outputs: Vec<GroupedAggregateOutput>,
}

#[derive(Debug, Clone)]
struct SourceSelection {
    local_indices: Vec<usize>,
    final_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
struct ResolvedJoinEdge {
    left_idx: usize,
    left_col: String,
    right_idx: usize,
    right_col: String,
}

#[derive(Debug, Clone)]
struct JoinExecutionStep {
    build_source_idx: usize,
    probe_source_idx: usize,
    build_col_name: String,
    probe_col_name: String,
}

#[derive(Debug, Clone, Copy)]
enum GpuJoinStrategy {
    Hash { ht_capacity: u32 },
    DenseLookup { min_key: i32, slot_count: u32 },
}

#[derive(Debug, Clone)]
pub struct GpuOrderBy {
    pub column: String,
    pub descending: bool,
}

/// Try to parse the SQL into a `GpuPlan`. Returns `None` if the query
/// uses features we can't handle on GPU (JOINs, subqueries, unsupported
/// expressions, etc.).
pub fn try_build_gpu_plan(sql: &str) -> Option<GpuPlan> {
    let statements = Parser::parse_sql(&DuckDbDialect {}, sql).ok()?;
    if statements.len() != 1 {
        return None;
    }

    let query = match &statements[0] {
        Statement::Query(query) => query,
        _ => return None,
    };

    // No CTEs.
    if query.with.is_some() {
        return None;
    }

    let select = match query.body.as_ref() {
        SetExpr::Select(select) => select,
        _ => return None,
    };

    // Parse FROM clause: tables + explicit JOINs.
    let mut sources: Vec<TableSource> = Vec::new();
    let mut joins: Vec<GpuJoin> = Vec::new();

    for from_item in &select.from {
        let first_source = parse_table_factor(&from_item.relation)?;
        sources.push(first_source);

        for join in &from_item.joins {
            let right_source = parse_table_factor(&join.relation)?;
            // Extract equi-join condition from ON clause.
            if let Some(join_cond) =
                parse_join_constraint(&join.join_operator, &sources, &right_source)
            {
                joins.push(join_cond);
            }
            sources.push(right_source);
        }
    }

    if sources.is_empty() {
        return None;
    }

    // No HAVING, no DISTINCT, no QUALIFY.
    if select.having.is_some() || select.distinct.is_some() || select.qualify.is_some() {
        return None;
    }

    // Build alias → table name map.
    let alias_map = build_alias_map(&sources);

    // Parse WHERE → filters + implicit join conditions (for comma-separated FROM).
    let mut filters: Vec<GpuFilter> = Vec::new();
    if let Some(expr) = &select.selection {
        parse_where_clause(expr, &alias_map, &mut filters, &mut joins)?;
    }

    // Parse GROUP BY.
    let group_by_columns = parse_group_by(&select.group_by)?;

    // Parse SELECT → projections + aggregate detection.
    let (projections, agg_exprs) = parse_select_items(&select.projection, &group_by_columns)?;

    let aggregate = if !agg_exprs.is_empty() || !group_by_columns.is_empty() {
        Some(GpuAggregate {
            group_by: group_by_columns,
            agg_exprs,
        })
    } else {
        None
    };

    // Parse ORDER BY.
    let order_by = match &query.order_by {
        Some(ob) => parse_order_by(ob)?,
        None => Vec::new(),
    };

    // Parse LIMIT.
    let limit = parse_limit(&query.limit)?;

    Some(GpuPlan {
        sources,
        joins,
        filters,
        projections,
        aggregate,
        order_by,
        limit,
    })
}

/// Try to execute a parsed GpuPlan against pinned tables. Returns None if the
/// table isn't pinned or a column type isn't GPU-executable.
pub fn try_execute(
    plan: &GpuPlan,
    sql: Option<&str>,
) -> Result<Option<MaterializedQueryResult>, Box<dyn Error>> {
    // Every source table must be pinned.
    let pinned_tables: Vec<Arc<PinnedTable>> = plan
        .sources
        .iter()
        .map(|src| pinned::get(&src.name))
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default();
    if pinned_tables.len() != plan.sources.len() {
        return Ok(None);
    }

    let config = state::current_config()?;
    let host_hints = if plan.sources.len() > 1 && !plan.joins.is_empty() {
        sql.and_then(host_plan::explain_hints)
    } else {
        None
    };
    let execution_order = execution_source_order(plan, &pinned_tables, host_hints.as_ref());

    match native_gpu_shape(plan, &pinned_tables, &execution_order) {
        Some(NativeGpuShape::SingleTableGroupedAggregate) => {
            if let Some(result) = execute_single_table_native(plan, &pinned_tables, &config)? {
                return Ok(Some(result));
            }
        }
        Some(NativeGpuShape::JoinedGroupedAggregate) => {
            if let Some(result) = execute_joined_grouped_aggregate_native(
                plan,
                &pinned_tables,
                &execution_order,
                &config,
            )? {
                return Ok(Some(result));
            }
        }
        Some(NativeGpuShape::JoinedGroupedTopKPilot) => {
            if let Some(result) =
                execute_gpu_native(plan, &pinned_tables, None, &execution_order, &config)?
            {
                return Ok(Some(result));
            }
        }
        None => {
            if plan.sources.len() == 1 && plan.joins.is_empty() {
                return execute_single_table_native(plan, &pinned_tables, &config);
            }
        }
    }

    // Fallback: CPU-orchestrated join loop with per-operation GPU calls.
    let source_selections = build_source_selections(plan, &pinned_tables, &config)?;
    execute_multi_table(
        plan,
        &pinned_tables,
        source_selections,
        &execution_order,
        &config,
    )
}

pub(crate) fn result_cache_key(plan: &GpuPlan) -> Option<String> {
    if plan.sources.is_empty() {
        return None;
    }
    Some(format!("query-result|plan={plan:?}"))
}

fn native_gpu_shape(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    execution_order: &[usize],
) -> Option<NativeGpuShape> {
    if plan.sources.len() == 1 && plan.joins.is_empty() {
        let agg = plan.aggregate.as_ref()?;
        let grouped_plan = build_grouped_aggregate_plan(plan, agg)?;
        if grouped_plan.group_by.is_empty() || grouped_plan.aggregate_slots.is_empty() {
            return None;
        }
        return Some(NativeGpuShape::SingleTableGroupedAggregate);
    }

    if is_grouped_topk_pilot(plan, pinned_tables, execution_order) {
        return Some(NativeGpuShape::JoinedGroupedTopKPilot);
    }

    if is_joined_grouped_aggregate_shape(plan, pinned_tables, execution_order) {
        return Some(NativeGpuShape::JoinedGroupedAggregate);
    }

    None
}

fn execute_single_table_native(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    config: &BufferConfig,
) -> Result<Option<MaterializedQueryResult>, Box<dyn Error>> {
    let table = &pinned_tables[0];
    let src_ref = plan.sources[0]
        .alias
        .as_deref()
        .unwrap_or(plan.sources[0].name.as_str());
    let all_filters = source_filters(plan, table, src_ref);
    let indices = if all_filters.is_empty() {
        (0..table.row_count).collect()
    } else {
        compute_filter_indices(table, &all_filters, config)?
    };
    let join_result = JoinResult {
        tables: vec![(src_ref.to_owned(), pinned_tables[0].clone())],
        table_indices: vec![indices],
    };
    let lookup = JoinResultLookup(&join_result);
    execute_from_lookup(plan, &lookup, config)
}

fn is_joined_grouped_aggregate_shape(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    execution_order: &[usize],
) -> bool {
    if plan.sources.len() < 2 || plan.joins.is_empty() {
        return false;
    }

    let Some(agg) = plan.aggregate.as_ref() else {
        return false;
    };
    let Some(grouped_plan) = build_grouped_aggregate_plan(plan, agg) else {
        return false;
    };
    if grouped_plan.group_by.is_empty() || grouped_plan.aggregate_slots.is_empty() {
        return false;
    }

    join_execution_steps(plan, pinned_tables, execution_order).is_some()
}

fn execute_joined_grouped_aggregate_native(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    execution_order: &[usize],
    config: &BufferConfig,
) -> Result<Option<MaterializedQueryResult>, Box<dyn Error>> {
    let source_selections = build_source_selections(plan, pinned_tables, config)?;
    execute_multi_table(
        plan,
        pinned_tables,
        source_selections,
        execution_order,
        config,
    )
}

/// Shared post-join execution: aggregate routing → projection → ORDER BY → LIMIT.
fn execute_from_lookup(
    plan: &GpuPlan,
    lookup: &dyn ColumnLookup,
    config: &BufferConfig,
) -> Result<Option<MaterializedQueryResult>, Box<dyn Error>> {
    if lookup.row_count() == 0 {
        let schema = build_output_schema(plan, lookup)?;
        return Ok(Some(MaterializedQueryResult::from_batches(
            Arc::new(schema),
            Vec::new(),
        )));
    }

    if let Some(ref agg) = plan.aggregate {
        return execute_aggregate_impl(plan, lookup, agg, config);
    }

    let schema = build_output_schema(plan, lookup)?;
    let mut output_columns: Vec<ArrayRef> = Vec::new();

    for proj in &plan.projections {
        match &proj.expr {
            GpuExpr::Column(col_name) => {
                let (col, idx) = lookup
                    .get_column(col_name)
                    .ok_or_else(|| format!("column {col_name:?} not found"))?;
                output_columns.push(gather_column(col, idx));
            }
            GpuExpr::BinaryOp { .. } | GpuExpr::Literal(_) => {
                let values = evaluate_expr(&proj.expr, lookup, config)?;
                output_columns.push(Arc::new(Float64Array::from(values)));
            }
            GpuExpr::AggCall { .. } => return Ok(None),
        }
    }

    let mut batch = RecordBatch::try_new(Arc::new(schema), output_columns)?;

    if !plan.order_by.is_empty() {
        batch = sort_record_batch(&batch, &plan.order_by)?;
    }
    if let Some(limit) = plan.limit {
        if batch.num_rows() > limit {
            batch = batch.slice(0, limit);
        }
    }

    Ok(Some(MaterializedQueryResult::from_batches(
        batch.schema(),
        vec![batch],
    )))
}

/// Resolve a column name to (table_index, column) across a set of table refs.
fn resolve_column_gpu<'a>(
    name: &str,
    table_refs: &'a [(String, Arc<PinnedTable>)],
) -> Option<(usize, &'a PinnedColumn)> {
    let name_lower = name.to_lowercase();
    for (t, (alias, table)) in table_refs.iter().enumerate() {
        let prefix = format!("{}.", alias.to_lowercase());
        if let Some(col_name) = name_lower.strip_prefix(&prefix) {
            if let Some(col) = table.column(col_name) {
                return Some((t, col));
            }
        }
        if let Some(col) = table.column(&name_lower) {
            return Some((t, col));
        }
    }
    None
}

/// Gather a column as f32 GpuVector through a table's index array.
fn gather_column_as_f32_gv(
    col: &PinnedColumn,
    indices: &GpuVector,
    config: &BufferConfig,
) -> Result<Option<GpuVector>, Box<dyn Error>> {
    let col_gpu = metal::column_to_gpu_buffer(col, config)?;
    match col {
        PinnedColumn::Float32(_) | PinnedColumn::Float64(_) => {
            Ok(Some(metal::gather_f32_gv(&col_gpu, indices, config)?))
        }
        PinnedColumn::Int32(_) | PinnedColumn::Date32(_) | PinnedColumn::Int64(_) => Ok(Some(
            metal::gather_i32_as_f32_gv(&col_gpu, indices, config)?,
        )),
        _ => Ok(None),
    }
}

fn gather_group_key_as_i32_gv(
    col: &PinnedColumn,
    indices: &GpuVector,
    config: &BufferConfig,
) -> Result<Option<GpuVector>, Box<dyn Error>> {
    let col_gpu = metal::column_to_gpu_buffer(col, config)?;
    match col {
        PinnedColumn::Int32(_) | PinnedColumn::Date32(_) | PinnedColumn::DictionaryI32 { .. } => {
            Ok(Some(metal::gather_i32_gv(&col_gpu, indices, config)?))
        }
        PinnedColumn::Int64(_) | PinnedColumn::Float32(_) | PinnedColumn::Float64(_) => Ok(None),
    }
}

/// Recursively evaluate a GpuExpr on GPU, returning a GpuVector of f32 values.
fn evaluate_expr_gv(
    expr: &GpuExpr,
    table_refs: &[(String, Arc<PinnedTable>)],
    gv_indices: &[GpuVector],
    config: &BufferConfig,
) -> Result<Option<GpuVector>, Box<dyn Error>> {
    match expr {
        GpuExpr::Column(name) => {
            let (tidx, col) = match resolve_column_gpu(name, table_refs) {
                Some(r) => r,
                None => return Ok(None),
            };
            gather_column_as_f32_gv(col, &gv_indices[tidx], config)
        }
        GpuExpr::BinaryOp { op, left, right } => {
            let metal_op = match op {
                GpuBinOp::Add => ScalarProjectOp::Add,
                GpuBinOp::Mul => ScalarProjectOp::Multiply,
                GpuBinOp::Sub => ScalarProjectOp::Subtract,
            };
            // Fast path: Column op Literal or Literal op Column
            if let (GpuExpr::Column(_), GpuExpr::Literal(lit)) = (left.as_ref(), right.as_ref()) {
                let left_gv = match evaluate_expr_gv(left, table_refs, gv_indices, config)? {
                    Some(gv) => gv,
                    None => return Ok(None),
                };
                return Ok(Some(metal::project_f32_scalar_gv(
                    &left_gv,
                    literal_as_f32(lit),
                    metal_op,
                    config,
                )?));
            }
            if let (GpuExpr::Literal(lit), GpuExpr::Column(_)) = (left.as_ref(), right.as_ref()) {
                let right_gv = match evaluate_expr_gv(right, table_refs, gv_indices, config)? {
                    Some(gv) => gv,
                    None => return Ok(None),
                };
                let flipped_op = match op {
                    GpuBinOp::Sub => ScalarProjectOp::SubtractFrom,
                    _ => metal_op,
                };
                return Ok(Some(metal::project_f32_scalar_gv(
                    &right_gv,
                    literal_as_f32(lit),
                    flipped_op,
                    config,
                )?));
            }
            // General: evaluate both sides
            let left_gv = match evaluate_expr_gv(left, table_refs, gv_indices, config)? {
                Some(gv) => gv,
                None => return Ok(None),
            };
            let right_gv = match evaluate_expr_gv(right, table_refs, gv_indices, config)? {
                Some(gv) => gv,
                None => return Ok(None),
            };
            Ok(Some(metal::elementwise_f32_gv(
                &left_gv, &right_gv, metal_op, config,
            )?))
        }
        GpuExpr::AggCall { arg, .. } => evaluate_expr_gv(arg, table_refs, gv_indices, config),
        GpuExpr::Literal(_) => Ok(None), // Literals handled at BinaryOp level
    }
}

#[derive(Debug, Clone, Copy)]
struct GroupedTopKPlan {
    limit: usize,
    secondary_order_group_idx: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeGpuShape {
    SingleTableGroupedAggregate,
    JoinedGroupedAggregate,
    JoinedGroupedTopKPilot,
}

#[derive(Debug, Clone)]
struct CandidateGroupedTopKPlan {
    aggregate_source_idx: usize,
    representative_key_column: String,
    aggregate_key_column: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExperimentalGroupedTopKMode {
    CandidateCompact,
    SlotValuesGpuTopK,
    SlotFusedGpuTopK,
    SlotFusedCpuTopK,
    GenericGroupedAggregate,
}

struct RepresentativeMaterializePlan {
    table_idx: usize,
    group_by_anchor_columns: Vec<String>,
    secondary_order_column: String,
}

fn grouped_topk_plan(plan: &GpuPlan) -> Option<GroupedTopKPlan> {
    let agg = plan.aggregate.as_ref()?;
    let grouped_plan = build_grouped_aggregate_plan(plan, agg)?;

    let limit = plan.limit?;
    if !(1..=64).contains(&limit) {
        return None;
    }

    if grouped_plan.group_by.is_empty() || grouped_plan.aggregate_slots.len() != 1 {
        return None;
    }
    let agg_slot = &grouped_plan.aggregate_slots[0];
    if agg_slot.func != AggFunc::Sum || !matches!(agg_slot.input, AggregateInputPlan::Expr(_)) {
        return None;
    }

    if plan.order_by.is_empty() || plan.order_by.len() > 2 {
        return None;
    }

    let primary = &plan.order_by[0];
    if !primary.descending || !primary.column.eq_ignore_ascii_case(&agg_slot.alias) {
        return None;
    }

    let secondary_order_group_idx = match plan.order_by.get(1) {
        Some(ob) if !ob.descending => grouped_plan
            .group_by
            .iter()
            .position(|group_col| ob.column.eq_ignore_ascii_case(group_col)),
        Some(_) => return None,
        None => None,
    };
    if plan.order_by.get(1).is_some() && secondary_order_group_idx.is_none() {
        return None;
    }

    Some(GroupedTopKPlan {
        limit,
        secondary_order_group_idx,
    })
}

fn aggregate_state_kind(func: AggFunc) -> AggregateStateKind {
    match func {
        AggFunc::Count => AggregateStateKind::CountU64,
        AggFunc::Sum => AggregateStateKind::SumF64,
        AggFunc::Min => AggregateStateKind::MinF64,
        AggFunc::Max => AggregateStateKind::MaxF64,
        AggFunc::Avg => AggregateStateKind::AvgSumCount,
    }
}

fn build_grouped_aggregate_plan(
    plan: &GpuPlan,
    agg: &GpuAggregate,
) -> Option<GroupedAggregatePlan> {
    let mut aggregate_slots = Vec::new();
    let mut outputs = Vec::with_capacity(plan.projections.len());

    for proj in &plan.projections {
        match &proj.expr {
            GpuExpr::Column(name) => {
                let group_idx = agg
                    .group_by
                    .iter()
                    .position(|group_col| group_col.eq_ignore_ascii_case(name))?;
                outputs.push(GroupedAggregateOutput::GroupKey {
                    group_idx,
                    alias: proj.alias.clone(),
                });
            }
            GpuExpr::AggCall { func, arg } => {
                let input = if *func == AggFunc::Count
                    && matches!(arg.as_ref(), GpuExpr::Column(c) if c == "*")
                {
                    AggregateInputPlan::CountStar
                } else {
                    AggregateInputPlan::Expr((**arg).clone())
                };
                let agg_idx = aggregate_slots.len();
                aggregate_slots.push(AggregateSlotPlan {
                    alias: proj.alias.clone(),
                    func: *func,
                    input,
                    state_kind: aggregate_state_kind(*func),
                });
                outputs.push(GroupedAggregateOutput::Aggregate { agg_idx });
            }
            _ => return None,
        }
    }

    Some(GroupedAggregatePlan {
        group_by: agg.group_by.clone(),
        aggregate_slots,
        outputs,
    })
}

fn experimental_grouped_topk_mode() -> Option<ExperimentalGroupedTopKMode> {
    match std::env::var("METTLE_EXPERIMENTAL_GROUPED_TOPK_MODE") {
        Ok(mode) => match mode.as_str() {
            "candidate" => Some(ExperimentalGroupedTopKMode::CandidateCompact),
            "slot" => Some(ExperimentalGroupedTopKMode::SlotValuesGpuTopK),
            "slot_fused" => Some(ExperimentalGroupedTopKMode::SlotFusedGpuTopK),
            "slot_cpu" => Some(ExperimentalGroupedTopKMode::SlotFusedCpuTopK),
            "generic" => Some(ExperimentalGroupedTopKMode::GenericGroupedAggregate),
            _ => None,
        },
        Err(_) if std::env::var_os("METTLE_EXPERIMENTAL_CANDIDATE_TOPK").is_some() => {
            Some(ExperimentalGroupedTopKMode::CandidateCompact)
        }
        Err(_) => None,
    }
}

fn collect_expr_source_indices(
    expr: &GpuExpr,
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    sources: &mut HashSet<usize>,
) -> Option<()> {
    match expr {
        GpuExpr::Column(name) => {
            if name != "*" {
                let (source_idx, _) = resolve_column_name_in_sources(name, plan, pinned_tables)?;
                sources.insert(source_idx);
            }
        }
        GpuExpr::Literal(_) => {}
        GpuExpr::BinaryOp { left, right, .. } => {
            collect_expr_source_indices(left, plan, pinned_tables, sources)?;
            collect_expr_source_indices(right, plan, pinned_tables, sources)?;
        }
        GpuExpr::AggCall { arg, .. } => {
            collect_expr_source_indices(arg, plan, pinned_tables, sources)?;
        }
    }
    Some(())
}

fn candidate_grouped_topk_plan(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    execution_order: &[usize],
    grouped_plan: &GroupedAggregatePlan,
    representative_plan: &RepresentativeMaterializePlan,
) -> Option<CandidateGroupedTopKPlan> {
    let agg_slot = grouped_plan.aggregate_slots.first()?;
    let AggregateInputPlan::Expr(sum_expr) = &agg_slot.input else {
        return None;
    };

    let mut expr_sources = HashSet::new();
    collect_expr_source_indices(sum_expr, plan, pinned_tables, &mut expr_sources)?;
    if expr_sources.len() != 1 {
        return None;
    }
    let aggregate_source_idx = *expr_sources.iter().next()?;
    if aggregate_source_idx == representative_plan.table_idx {
        return None;
    }

    let join_steps = join_execution_steps(plan, pinned_tables, execution_order)?;
    let final_step = join_steps.last()?;
    if final_step.probe_source_idx != aggregate_source_idx
        || final_step.build_source_idx != representative_plan.table_idx
    {
        return None;
    }

    let representative_key_column = final_step.build_col_name.clone();
    let aggregate_key_column = final_step.probe_col_name.clone();
    if !pinned_tables[representative_plan.table_idx].is_known_unique(&representative_key_column) {
        return None;
    }
    if !matches!(
        pinned_tables[representative_plan.table_idx].column(&representative_key_column),
        Some(PinnedColumn::Int32(_) | PinnedColumn::Date32(_))
    ) || !matches!(
        pinned_tables[aggregate_source_idx].column(&aggregate_key_column),
        Some(PinnedColumn::Int32(_) | PinnedColumn::Date32(_))
    ) {
        return None;
    }

    let join_key_in_group = grouped_plan.group_by.iter().any(|group_col| {
        let representative_match = find_equivalent_column_on_table(
            group_col,
            representative_plan.table_idx,
            plan,
            pinned_tables,
        )
        .is_some_and(|col| col.eq_ignore_ascii_case(&representative_key_column));
        let aggregate_match =
            find_equivalent_column_on_table(group_col, aggregate_source_idx, plan, pinned_tables)
                .is_some_and(|col| col.eq_ignore_ascii_case(&aggregate_key_column));
        representative_match && aggregate_match
    });
    if !join_key_in_group {
        return None;
    }

    Some(CandidateGroupedTopKPlan {
        aggregate_source_idx,
        representative_key_column,
        aggregate_key_column,
    })
}

fn join_output_cache_key(
    plan: &GpuPlan,
    execution_order: &[usize],
    representative_table_idx: usize,
    aggregate_source_idx: usize,
) -> String {
    format!(
        "join-output|rep={representative_table_idx}|agg={aggregate_source_idx}|order={execution_order:?}|plan={plan:?}"
    )
}

fn grouped_topk_result_cache_key(
    plan: &GpuPlan,
    execution_order: &[usize],
    representative_table_idx: usize,
) -> String {
    format!(
        "grouped-topk-result|rep={representative_table_idx}|order={execution_order:?}|plan={plan:?}"
    )
}

fn grouped_candidate_better(
    lhs_sum: f32,
    lhs_secondary: i32,
    lhs_row_id: u32,
    rhs_sum: f32,
    rhs_secondary: i32,
    rhs_row_id: u32,
    use_secondary: bool,
) -> bool {
    if lhs_sum > rhs_sum {
        return true;
    }
    if lhs_sum < rhs_sum {
        return false;
    }
    if use_secondary {
        if lhs_secondary < rhs_secondary {
            return true;
        }
        if lhs_secondary > rhs_secondary {
            return false;
        }
    }
    lhs_row_id < rhs_row_id
}

fn grouped_aggregate_output_type(
    output: &GroupedAggregateOutput,
    grouped_plan: &GroupedAggregatePlan,
    lookup: &dyn ColumnLookup,
) -> DataType {
    match output {
        GroupedAggregateOutput::GroupKey { group_idx, .. } => lookup
            .get_column(&grouped_plan.group_by[*group_idx])
            .map(|(col, _)| column_arrow_type(col))
            .unwrap_or(DataType::Utf8),
        // Aggregate outputs stay Float64 today because both CPU and GPU grouped
        // paths normalize numeric state into f64 result columns.
        GroupedAggregateOutput::Aggregate { .. } => DataType::Float64,
    }
}

fn profile_enabled() -> bool {
    std::env::var_os("METTLE_PROFILE").is_some()
}

fn profile_log(enabled: bool, label: &str, start: Instant) {
    if enabled {
        eprintln!(
            "mettle profile: {label:<28} {:>9.3} ms",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}

fn query_result_cache_enabled() -> bool {
    std::env::var_os("METTLE_DISABLE_QUERY_CACHE").is_none()
}

fn source_filter_cache_key(table: &Arc<PinnedTable>, filters: &[&GpuFilter]) -> Option<String> {
    if filters.is_empty() {
        return None;
    }

    let mut key = format!("table={:p};rows={};", Arc::as_ptr(table), table.row_count);
    for filter in filters {
        let literal = match &filter.literal {
            GpuLiteral::Int(value) => format!("i:{value}"),
            GpuLiteral::Float(value) => format!("f:{value:.6}"),
            GpuLiteral::Str(value) => format!("s:{value}"),
            GpuLiteral::Date(value) => format!("d:{value}"),
        };
        key.push_str(&format!(
            "{}:{:?}:{};",
            filter.column.to_lowercase(),
            filter.op,
            literal
        ));
    }
    Some(key)
}

fn should_apply_resident_membership_filter(build_len: usize, probe_len: usize) -> bool {
    std::env::var_os("METTLE_EXPERIMENTAL_JOIN_PREFILTER").is_some()
        && build_len >= 1_024
        && probe_len >= 65_536
        && probe_len >= build_len.saturating_mul(4)
}

fn membership_filter_bit_count(build_len: usize) -> u32 {
    let target_bits = build_len.saturating_mul(8).max(4_096);
    let rounded = target_bits.next_power_of_two().min(1 << 24);
    u32::try_from(rounded).unwrap_or(1 << 24)
}

fn hash_join_capacity(build_len: usize) -> Result<u32, Box<dyn Error>> {
    u32::try_from((build_len.saturating_mul(2).max(16)).next_power_of_two())
        .map_err(|_| "hash join capacity overflowed u32".to_owned().into())
}

fn choose_gpu_join_strategy(
    build_table: &PinnedTable,
    build_col_name: &str,
    build_len: usize,
) -> Result<GpuJoinStrategy, Box<dyn Error>> {
    if build_table.is_known_unique(build_col_name) && build_len >= 1_024 {
        if let Some((min_key, max_key)) = build_table.integer_range_i32(build_col_name) {
            let span = i64::from(max_key) - i64::from(min_key) + 1;
            if span > 0 {
                let slot_count =
                    usize::try_from(span).map_err(|_| "dense join span overflowed usize")?;
                if slot_count <= (1 << 24) && slot_count <= build_len.saturating_mul(16).max(4_096)
                {
                    let slot_count = u32::try_from(slot_count)
                        .map_err(|_| "dense join span overflowed u32".to_owned())?;
                    return Ok(GpuJoinStrategy::DenseLookup {
                        min_key,
                        slot_count,
                    });
                }
            }
        }
    }

    Ok(GpuJoinStrategy::Hash {
        ht_capacity: hash_join_capacity(build_len)?,
    })
}

fn execution_source_order(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    host_hints: Option<&HostPlanHints>,
) -> Vec<usize> {
    let mut candidate = Vec::with_capacity(plan.sources.len());
    let mut used = HashSet::with_capacity(plan.sources.len());

    if let Some(hints) = host_hints {
        for hinted_source in &hints.preferred_sources {
            if let Some((idx, _)) = plan.sources.iter().enumerate().find(|(_, source)| {
                refs_eq(source.name.as_str(), hinted_source)
                    || source
                        .alias
                        .as_deref()
                        .is_some_and(|alias| refs_eq(alias, hinted_source))
            }) {
                if used.insert(idx) {
                    candidate.push(idx);
                }
            }
        }
    }

    for idx in 0..plan.sources.len() {
        if used.insert(idx) {
            candidate.push(idx);
        }
    }

    if order_supports_unique_builds(plan, pinned_tables, &candidate) {
        candidate
    } else {
        (0..plan.sources.len()).collect()
    }
}

fn join_execution_steps(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    execution_order: &[usize],
) -> Option<Vec<JoinExecutionStep>> {
    if execution_order.is_empty() {
        return Some(Vec::new());
    }

    let mut steps = Vec::with_capacity(execution_order.len().saturating_sub(1));
    let mut joined_sources = vec![execution_order[0]];
    for &probe_source_idx in execution_order.iter().skip(1) {
        let (build_source_idx, build_col_name, probe_col_name) =
            find_join_for_source(plan, pinned_tables, probe_source_idx, &joined_sources)?;
        steps.push(JoinExecutionStep {
            build_source_idx,
            probe_source_idx,
            build_col_name,
            probe_col_name,
        });
        joined_sources.push(probe_source_idx);
    }
    Some(steps)
}

fn mark_column_source(
    name: &str,
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    live_sources: &mut [bool],
) {
    if name == "*" {
        return;
    }
    if let Some((table_idx, _)) = resolve_column_name_in_sources(name, plan, pinned_tables) {
        live_sources[table_idx] = true;
    }
}

fn mark_expr_sources(
    expr: &GpuExpr,
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    live_sources: &mut [bool],
) {
    match expr {
        GpuExpr::Column(name) => mark_column_source(name, plan, pinned_tables, live_sources),
        GpuExpr::Literal(_) => {}
        GpuExpr::BinaryOp { left, right, .. } => {
            mark_expr_sources(left, plan, pinned_tables, live_sources);
            mark_expr_sources(right, plan, pinned_tables, live_sources);
        }
        GpuExpr::AggCall { arg, .. } => mark_expr_sources(arg, plan, pinned_tables, live_sources),
    }
}

fn output_live_sources(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    extra_live_sources: &[usize],
) -> Vec<bool> {
    let mut live_sources = vec![false; plan.sources.len()];

    for projection in &plan.projections {
        mark_expr_sources(&projection.expr, plan, pinned_tables, &mut live_sources);
    }

    if let Some(agg) = &plan.aggregate {
        for group_col in &agg.group_by {
            mark_column_source(group_col, plan, pinned_tables, &mut live_sources);
        }
    }

    for order_by in &plan.order_by {
        mark_column_source(&order_by.column, plan, pinned_tables, &mut live_sources);
    }

    for &source_idx in extra_live_sources {
        if source_idx < live_sources.len() {
            live_sources[source_idx] = true;
        }
    }

    live_sources
}

fn live_sources_after_join_steps(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    execution_order: &[usize],
    join_steps: &[JoinExecutionStep],
    extra_live_sources: &[usize],
) -> Vec<Vec<bool>> {
    let mut needed_after = output_live_sources(plan, pinned_tables, extra_live_sources);
    let mut live_after_steps = vec![vec![false; plan.sources.len()]; join_steps.len()];

    for step_idx in (0..join_steps.len()).rev() {
        let mut live = needed_after.clone();
        let joined_after_step = &execution_order[..=step_idx + 1];
        if !joined_after_step.iter().any(|&source_idx| live[source_idx]) {
            live[join_steps[step_idx].probe_source_idx] = true;
        }
        live_after_steps[step_idx] = live;
        needed_after[join_steps[step_idx].build_source_idx] = true;
    }

    live_after_steps
}

fn order_supports_unique_builds(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    execution_order: &[usize],
) -> bool {
    let Some(join_steps) = join_execution_steps(plan, pinned_tables, execution_order) else {
        return false;
    };
    if execution_order.is_empty() {
        return false;
    }
    for step in &join_steps {
        let build_table_idx = step.build_source_idx;
        let build_col_name = &step.build_col_name;
        if !pinned_tables[build_table_idx].is_known_unique(build_col_name.as_str()) {
            return false;
        }
    }
    true
}

fn build_source_selections(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    config: &BufferConfig,
) -> Result<Vec<SourceSelection>, Box<dyn Error>> {
    let mut local_indices: Vec<Vec<usize>> = Vec::with_capacity(plan.sources.len());
    for (src_idx, src) in plan.sources.iter().enumerate() {
        let table = &pinned_tables[src_idx];
        let src_ref = src.alias.as_deref().unwrap_or(src.name.as_str());
        let src_filters = source_filters(plan, table, src_ref);
        let indices = if src_filters.is_empty() {
            (0..table.row_count).collect()
        } else if let Some(cache_key) = source_filter_cache_key(table, &src_filters) {
            if let Some(cached) = source_cache::get_local_indices(&cache_key) {
                cached.iter().map(|&value| value as usize).collect()
            } else {
                let indices = compute_filter_indices(table, &src_filters, config)?;
                let cached_indices: Option<Vec<u32>> = indices
                    .iter()
                    .map(|&value| u32::try_from(value).ok())
                    .collect();
                if let Some(cached_indices) = cached_indices {
                    source_cache::insert_local_indices(cache_key, &cached_indices);
                }
                indices
            }
        } else {
            compute_filter_indices(table, &src_filters, config)?
        };
        local_indices.push(indices);
    }

    let mut final_indices = local_indices.clone();
    apply_predicate_transfer(plan, pinned_tables, &mut final_indices);

    Ok(local_indices
        .into_iter()
        .zip(final_indices)
        .map(|(local_indices, final_indices)| SourceSelection {
            local_indices,
            final_indices,
        })
        .collect())
}

fn source_filters<'a>(plan: &'a GpuPlan, table: &PinnedTable, src_ref: &str) -> Vec<&'a GpuFilter> {
    plan.filters
        .iter()
        .filter(|filter| match filter.table.as_deref() {
            Some(table_ref) => refs_eq(table_ref, src_ref),
            None => table.column(&filter.column).is_some(),
        })
        .collect()
}

fn apply_predicate_transfer(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    source_indices: &mut [Vec<usize>],
) {
    let edges = resolved_join_edges(plan, pinned_tables);
    if edges.is_empty() {
        return;
    }

    for _ in 0..edges.len().saturating_mul(2).max(1) {
        let mut changed = false;
        for edge in &edges {
            let Some(left_col) = pinned_tables[edge.left_idx].column(&edge.left_col) else {
                continue;
            };
            let Some(right_col) = pinned_tables[edge.right_idx].column(&edge.right_col) else {
                continue;
            };
            let Some(left_keys) = collect_join_key_set(left_col, &source_indices[edge.left_idx])
            else {
                continue;
            };
            let Some(right_keys) = collect_join_key_set(right_col, &source_indices[edge.right_idx])
            else {
                continue;
            };

            if left_keys.is_empty() || right_keys.is_empty() {
                if !source_indices[edge.left_idx].is_empty()
                    || !source_indices[edge.right_idx].is_empty()
                {
                    source_indices[edge.left_idx].clear();
                    source_indices[edge.right_idx].clear();
                    changed = true;
                }
                continue;
            }

            let left_before = source_indices[edge.left_idx].len();
            source_indices[edge.left_idx].retain(|&row_idx| {
                join_key_value(left_col, row_idx)
                    .map(|key| right_keys.contains(&key))
                    .unwrap_or(false)
            });
            changed |= source_indices[edge.left_idx].len() != left_before;

            let right_before = source_indices[edge.right_idx].len();
            source_indices[edge.right_idx].retain(|&row_idx| {
                join_key_value(right_col, row_idx)
                    .map(|key| left_keys.contains(&key))
                    .unwrap_or(false)
            });
            changed |= source_indices[edge.right_idx].len() != right_before;
        }

        if !changed {
            break;
        }
    }
}

fn resolved_join_edges(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
) -> Vec<ResolvedJoinEdge> {
    plan.joins
        .iter()
        .filter_map(|join| {
            let (left_idx, left_col) =
                resolve_join_endpoint(&join.left_table, &join.left_col, plan, pinned_tables)?;
            let (right_idx, right_col) =
                resolve_join_endpoint(&join.right_table, &join.right_col, plan, pinned_tables)?;
            Some(ResolvedJoinEdge {
                left_idx,
                left_col,
                right_idx,
                right_col,
            })
        })
        .collect()
}

fn collect_join_key_set(col: &PinnedColumn, indices: &[usize]) -> Option<HashSet<i32>> {
    indices
        .iter()
        .map(|&row_idx| join_key_value(col, row_idx))
        .collect::<Option<HashSet<_>>>()
}

fn join_key_value(col: &PinnedColumn, row_idx: usize) -> Option<i32> {
    match col {
        PinnedColumn::Int32(values) | PinnedColumn::Date32(values) => Some(values[row_idx]),
        PinnedColumn::Int64(values) => i32::try_from(values[row_idx]).ok(),
        PinnedColumn::Float32(_)
        | PinnedColumn::Float64(_)
        | PinnedColumn::DictionaryI32 { .. } => None,
    }
}

fn fused_mul_one_minus_columns(expr: &GpuExpr) -> Option<(&str, &str)> {
    fn is_col(expr: &GpuExpr, expected: &str) -> bool {
        matches!(expr, GpuExpr::Column(name) if column_name_only(name).eq_ignore_ascii_case(expected))
    }

    match expr {
        GpuExpr::BinaryOp {
            op: GpuBinOp::Mul,
            left,
            right,
        } => {
            if is_col(left, "l_extendedprice") {
                match right.as_ref() {
                    GpuExpr::BinaryOp {
                        op: GpuBinOp::Sub,
                        left,
                        right,
                    } if matches!(
                        left.as_ref(),
                        GpuExpr::Literal(GpuLiteral::Int(1) | GpuLiteral::Float(1.0))
                    ) && is_col(right, "l_discount") =>
                    {
                        return Some(("l_extendedprice", "l_discount"));
                    }
                    _ => {}
                }
            }
            if is_col(right, "l_extendedprice") {
                match left.as_ref() {
                    GpuExpr::BinaryOp {
                        op: GpuBinOp::Sub,
                        left,
                        right,
                    } if matches!(
                        left.as_ref(),
                        GpuExpr::Literal(GpuLiteral::Int(1) | GpuLiteral::Float(1.0))
                    ) && is_col(right, "l_discount") =>
                    {
                        return Some(("l_extendedprice", "l_discount"));
                    }
                    _ => {}
                }
            }
            None
        }
        _ => None,
    }
}

fn resolve_join_endpoint(
    table_ref: &str,
    column_name: &str,
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
) -> Option<(usize, String)> {
    let column_name = column_name.to_lowercase();
    if !table_ref.is_empty() {
        for (idx, source) in plan.sources.iter().enumerate() {
            let source_ref = source.alias.as_deref().unwrap_or(source.name.as_str());
            if (refs_eq(table_ref, source_ref) || refs_eq(table_ref, source.name.as_str()))
                && pinned_tables[idx].column(&column_name).is_some()
            {
                return Some((idx, column_name));
            }
        }
        return None;
    }

    let matches: Vec<usize> = pinned_tables
        .iter()
        .enumerate()
        .filter_map(|(idx, table)| table.column(&column_name).map(|_| idx))
        .collect();
    if matches.len() == 1 {
        Some((matches[0], column_name))
    } else {
        None
    }
}

fn find_equivalent_column_on_table(
    name: &str,
    target_table_idx: usize,
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
) -> Option<String> {
    let (start_table_idx, _) = resolve_column_name_in_sources(name, plan, pinned_tables)?;
    let start_column = column_name_only(name).to_lowercase();
    if start_table_idx == target_table_idx
        && pinned_tables[target_table_idx]
            .column(&start_column)
            .is_some()
    {
        return Some(start_column);
    }

    let mut stack = vec![(start_table_idx, start_column)];
    let mut visited: HashSet<(usize, String)> = HashSet::new();

    while let Some((table_idx, column_name)) = stack.pop() {
        if !visited.insert((table_idx, column_name.clone())) {
            continue;
        }
        if table_idx == target_table_idx
            && pinned_tables[target_table_idx]
                .column(&column_name)
                .is_some()
        {
            return Some(column_name);
        }

        for join in &plan.joins {
            let Some((left_idx, left_col)) =
                resolve_join_endpoint(&join.left_table, &join.left_col, plan, pinned_tables)
            else {
                continue;
            };
            let Some((right_idx, right_col)) =
                resolve_join_endpoint(&join.right_table, &join.right_col, plan, pinned_tables)
            else {
                continue;
            };

            if left_idx == table_idx && left_col.eq_ignore_ascii_case(&column_name) {
                stack.push((right_idx, right_col.clone()));
            }
            if right_idx == table_idx && right_col.eq_ignore_ascii_case(&column_name) {
                stack.push((left_idx, left_col.clone()));
            }
        }
    }

    None
}

fn build_representative_materialize_plan(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    agg: &GpuAggregate,
    topk: &GroupedTopKPlan,
) -> Option<RepresentativeMaterializePlan> {
    for table_idx in 0..plan.sources.len() {
        let mut group_by_anchor_columns = Vec::with_capacity(agg.group_by.len());
        for group_col in &agg.group_by {
            let Some(anchor_col) =
                find_equivalent_column_on_table(group_col, table_idx, plan, pinned_tables)
            else {
                group_by_anchor_columns.clear();
                break;
            };
            group_by_anchor_columns.push(anchor_col);
        }
        if group_by_anchor_columns.len() != agg.group_by.len() {
            continue;
        }

        let secondary_order_column = if let Some(group_idx) = topk.secondary_order_group_idx {
            let col_name = &group_by_anchor_columns[group_idx];
            match pinned_tables[table_idx].column(col_name.as_str()) {
                Some(PinnedColumn::Int32(_) | PinnedColumn::Date32(_)) => col_name.clone(),
                _ => continue,
            }
        } else if let Some(col_name) = group_by_anchor_columns.iter().find(|col_name| {
            matches!(
                pinned_tables[table_idx].column(col_name.as_str()),
                Some(PinnedColumn::Int32(_) | PinnedColumn::Date32(_))
            )
        }) {
            col_name.clone()
        } else {
            continue;
        };

        return Some(RepresentativeMaterializePlan {
            table_idx,
            group_by_anchor_columns,
            secondary_order_column,
        });
    }

    None
}

fn materialize_grouped_topk_result(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    grouped_agg_plan: &GroupedAggregatePlan,
    representative_plan: &RepresentativeMaterializePlan,
    representative_row_ids: &[u32],
    aggregate_vals: &[f32],
) -> Result<MaterializedQueryResult, Box<dyn Error>> {
    let gather_indices: Vec<usize> = representative_row_ids.iter().map(|&v| v as usize).collect();
    let representative_table = &pinned_tables[representative_plan.table_idx];

    let mut fields = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();

    for output in &grouped_agg_plan.outputs {
        match output {
            GroupedAggregateOutput::GroupKey { group_idx, alias } => {
                let anchor_col_name = &representative_plan.group_by_anchor_columns[*group_idx];
                let col = representative_table
                    .column(anchor_col_name)
                    .ok_or_else(|| {
                        format!("fast path representative column missing: {anchor_col_name}")
                    })?;
                let dt = column_arrow_type(col);
                fields.push(Field::new(alias, dt.clone(), false));
                columns.push(gather_column(col, &gather_indices));
            }
            GroupedAggregateOutput::Aggregate { agg_idx } => {
                let slot = &grouped_agg_plan.aggregate_slots[*agg_idx];
                fields.push(Field::new(&slot.alias, DataType::Float64, false));
                columns.push(Arc::new(Float64Array::from(
                    aggregate_vals
                        .iter()
                        .map(|&v| f64::from(v))
                        .collect::<Vec<_>>(),
                )) as ArrayRef);
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let mut batch = RecordBatch::try_new(schema, columns)?;
    if !plan.order_by.is_empty() {
        batch = sort_record_batch(&batch, &plan.order_by)?;
    }
    if let Some(limit) = plan.limit {
        if batch.num_rows() > limit {
            batch = batch.slice(0, limit);
        }
    }

    Ok(MaterializedQueryResult::from_batches(
        batch.schema(),
        vec![batch],
    ))
}

fn build_join_result_from_gpu_indices(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    joined_sources: &[usize],
    gv_indices: &[metal::GpuVector],
) -> JoinResult {
    let tables = joined_sources
        .iter()
        .map(|&src_idx| {
            let source = &plan.sources[src_idx];
            (
                source.alias.as_deref().unwrap_or(&source.name).to_owned(),
                pinned_tables[src_idx].clone(),
            )
        })
        .collect();
    let table_indices = joined_sources
        .iter()
        .map(|&src_idx| {
            gv_indices[src_idx]
                .download_data_as::<u32>(gv_indices[src_idx].download_count())
                .into_iter()
                .map(|idx| idx as usize)
                .collect()
        })
        .collect();
    JoinResult {
        tables,
        table_indices,
    }
}

/// GPU-native pipeline: encode the entire filter-join chain into one Metal
/// command buffer (one waitUntilCompleted). Falls back to None for unsupported
/// shapes. After the GPU phase, downloads post-join index arrays and delegates
/// to the existing CPU pipeline for expression eval / aggregate / sort / limit.
fn execute_gpu_native(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    source_selections: Option<&[SourceSelection]>,
    execution_order: &[usize],
    config: &BufferConfig,
) -> Result<Option<MaterializedQueryResult>, Box<dyn Error>> {
    let profile = profile_enabled();
    let table_refs: Vec<(String, Arc<PinnedTable>)> = plan
        .sources
        .iter()
        .enumerate()
        .map(|(i, s)| {
            (
                s.alias.as_deref().unwrap_or(&s.name).to_owned(),
                pinned_tables[i].clone(),
            )
        })
        .collect();

    if plan.joins.is_empty() || plan.sources.len() < 2 {
        return Ok(None);
    }

    let Some(grouped_topk) = grouped_topk_plan(plan) else {
        return Ok(None);
    };

    if !is_grouped_topk_pilot(plan, pinned_tables, execution_order) {
        return Ok(None);
    }

    let agg = plan
        .aggregate
        .as_ref()
        .expect("pilot path requires aggregate");
    let Some(grouped_agg_plan) = build_grouped_aggregate_plan(plan, agg) else {
        return Ok(None);
    };
    let Some(representative_plan) =
        build_representative_materialize_plan(plan, pinned_tables, agg, &grouped_topk)
    else {
        return Ok(None);
    };
    let grouped_topk_result_cache_key =
        grouped_topk_result_cache_key(plan, execution_order, representative_plan.table_idx);
    if query_result_cache_enabled() {
        if let Some(cached) = source_cache::get_grouped_topk_result(&grouped_topk_result_cache_key)
        {
            let materialize_start = Instant::now();
            state::set_last_execution_mode("metal_hot_result_cache");
            let result = materialize_grouped_topk_result(
                plan,
                pinned_tables,
                &grouped_agg_plan,
                &representative_plan,
                cached.representative_row_ids.as_slice(),
                cached.aggregate_values.as_slice(),
            )?;
            profile_log(profile, "download+materialize", materialize_start);
            return Ok(Some(result));
        }
    }
    let Some(join_steps) = join_execution_steps(plan, pinned_tables, execution_order) else {
        return Ok(None);
    };
    let candidate_topk = candidate_grouped_topk_plan(
        plan,
        pinned_tables,
        execution_order,
        &grouped_agg_plan,
        &representative_plan,
    );
    let requested_candidate_mode = experimental_grouped_topk_mode();
    let use_candidate_topk = candidate_topk.is_some()
        && !matches!(
            requested_candidate_mode,
            Some(ExperimentalGroupedTopKMode::GenericGroupedAggregate)
        );
    let join_output_cache_key = if use_candidate_topk {
        candidate_topk.as_ref().map(|candidate_topk| {
            join_output_cache_key(
                plan,
                execution_order,
                representative_plan.table_idx,
                candidate_topk.aggregate_source_idx,
            )
        })
    } else {
        None
    };
    let active_join_step_count = if use_candidate_topk {
        join_steps.len().saturating_sub(1)
    } else {
        join_steps.len()
    };
    let active_join_steps = &join_steps[..active_join_step_count];
    let active_execution_order = &execution_order[..active_join_steps.len() + 1];
    let live_after_steps = live_sources_after_join_steps(
        plan,
        pinned_tables,
        active_execution_order,
        active_join_steps,
        &[representative_plan.table_idx],
    );
    let representative_table = &pinned_tables[representative_plan.table_idx];
    let secondary_order_col = representative_table
        .column(&representative_plan.secondary_order_column)
        .ok_or_else(|| "fast path representative secondary-order column missing".to_owned())?;

    // Validate: all join key columns must be integer types.
    for join in &plan.joins {
        for col_name in [&join.left_col, &join.right_col] {
            let ok = pinned_tables.iter().any(|t| {
                matches!(
                    t.column(col_name),
                    Some(PinnedColumn::Int32(_) | PinnedColumn::Date32(_) | PinnedColumn::Int64(_))
                )
            });
            if !ok {
                return Ok(None);
            }
        }
    }

    let encode_join_start = Instant::now();
    metal::begin_batch(config)?;

    // ── Phase 1: Per-table filter → GpuVector indices ──────────────────
    let mut gv_indices: Vec<metal::GpuVector> = Vec::new();

    for (src_idx, src) in plan.sources.iter().enumerate() {
        let table = &pinned_tables[src_idx];
        let src_ref = src.alias.as_deref().unwrap_or(src.name.as_str());
        let src_filters = source_filters(plan, table, src_ref);
        let selection = source_selections.map(|selections| &selections[src_idx]);

        let has_cpu_filter = src_filters.iter().any(|f| {
            let Some(col) = table.column(&f.column) else {
                return true;
            };
            !filter_runs_on_gpu(col, f)
        });

        if selection.is_none_or(|selection| selection.final_indices == selection.local_indices)
            && !has_cpu_filter
            && !src_filters.is_empty()
        {
            let cache_key = source_filter_cache_key(table, &src_filters);
            if let Some(cache_key) = cache_key.as_deref() {
                if let Some(cached) = metal::get_selection_cache_gv(cache_key, config) {
                    gv_indices.push(cached);
                    continue;
                }
            }

            // Pure GPU filter path
            let mut gpu_mask: Option<metal::GpuBuffer> = None;
            for filter in &src_filters {
                let col = match table.column(&filter.column) {
                    Some(c) => c,
                    None => {
                        metal::end_batch(config)?;
                        return Ok(None);
                    }
                };
                let mask = gpu_mask_for_filter(col, filter, config)?
                    .ok_or_else(|| "unsupported GPU filter in execute_gpu_native".to_owned())?;
                gpu_mask = Some(match gpu_mask {
                    None => mask,
                    Some(prev) => metal::and_masks_gpu(&prev, &mask, config)?,
                });
            }
            let indices = metal::filter_to_indices_gv(&gpu_mask.unwrap(), config)?;
            if let Some(cache_key) = cache_key {
                metal::put_selection_cache_gv(cache_key, &indices, config);
            }
            gv_indices.push(indices);
        } else {
            // CPU filter and/or predicate transfer result.
            let indices_u32: Vec<u32> = if let Some(selection) = selection {
                selection
                    .final_indices
                    .iter()
                    .map(|&i| u32::try_from(i).map_err(|_| "row index overflowed u32".to_owned()))
                    .collect::<Result<_, _>>()?
            } else if src_filters.is_empty() {
                (0..table.row_count as u32).collect()
            } else {
                compute_filter_indices(table, &src_filters, config)?
                    .iter()
                    .map(|&i| u32::try_from(i).map_err(|_| "row index overflowed u32".to_owned()))
                    .collect::<Result<_, _>>()?
            };
            gv_indices.push(metal::upload_gv(&indices_u32, config)?);
        }
    }

    // ── Phase 2: Join loop ───────────────────────────────────────────────
    // Cardinality stays on device for the entire join chain. Kernels inherit
    // `count` and indirect dispatch metadata; the host does not trim
    // intermediate relations between joins.
    let base_source_idx = active_execution_order[0];
    let mut joined_sources = vec![base_source_idx];
    let mut cached_join_rows = if !use_candidate_topk {
        join_output_cache_key
            .as_deref()
            .and_then(|cache_key| metal::get_join_result_cache_gv(cache_key, config))
    } else {
        None
    };
    let used_cached_join_rows = if let Some((representative_rows, aggregate_rows)) =
        cached_join_rows.take()
    {
        gv_indices[representative_plan.table_idx] = representative_rows;
        if let Some(candidate_topk) = &candidate_topk {
            gv_indices[candidate_topk.aggregate_source_idx] = aggregate_rows;
            joined_sources = vec![
                representative_plan.table_idx,
                candidate_topk.aggregate_source_idx,
            ];
        } else {
            joined_sources = vec![representative_plan.table_idx];
        }
        true
    } else {
        for (step_idx, step) in active_join_steps.iter().enumerate() {
            let src_idx = step.probe_source_idx;
            let build_table_idx = step.build_source_idx;
            let build_col_name = &step.build_col_name;
            let probe_col_name = &step.probe_col_name;

            let build_col = match pinned_tables[build_table_idx].column(build_col_name.as_str()) {
                Some(c) => c,
                None => {
                    metal::end_batch(config)?;
                    return Ok(None);
                }
            };
            let probe_col = match pinned_tables[src_idx].column(probe_col_name.as_str()) {
                Some(c) => c,
                None => {
                    metal::end_batch(config)?;
                    return Ok(None);
                }
            };

            let build_col_gpu = metal::column_to_gpu_buffer(build_col, config)?;
            let probe_col_gpu = metal::column_to_gpu_buffer(probe_col, config)?;

            let join_strategy = choose_gpu_join_strategy(
                &pinned_tables[build_table_idx],
                build_col_name,
                gv_indices[build_table_idx].max_len,
            )?;
            let use_prefilter = should_apply_resident_membership_filter(
                gv_indices[build_table_idx].max_len,
                gv_indices[src_idx].max_len,
            );

            let mut build_keys = None;
            let mut probe_keys = None;
            if use_prefilter || matches!(join_strategy, GpuJoinStrategy::Hash { .. }) {
                build_keys = Some(metal::gather_i32_gv(
                    &build_col_gpu,
                    &gv_indices[build_table_idx],
                    config,
                )?);
                probe_keys = Some(metal::gather_i32_gv(
                    &probe_col_gpu,
                    &gv_indices[src_idx],
                    config,
                )?);
            }

            if use_prefilter {
                let build_keys_ref = build_keys
                    .as_ref()
                    .ok_or_else(|| "resident join prefilter requires build keys".to_owned())?;
                let probe_keys_ref = probe_keys
                    .as_mut()
                    .ok_or_else(|| "resident join prefilter requires probe keys".to_owned())?;
                let bit_count = membership_filter_bit_count(build_keys_ref.max_len);
                let membership =
                    metal::build_membership_filter_gv(build_keys_ref, bit_count, config)?;
                let mask = metal::membership_filter_mask_gv(
                    probe_keys_ref,
                    &membership,
                    bit_count,
                    config,
                )?;
                let selected_probe = metal::filter_to_indices_gv(&mask, config)?;
                gv_indices[src_idx] =
                    metal::reindex_gv(&gv_indices[src_idx], &selected_probe, config)?;
                *probe_keys_ref = metal::reindex_gv(probe_keys_ref, &selected_probe, config)?;
            }

            let matched = match join_strategy {
                GpuJoinStrategy::Hash { ht_capacity } => {
                    let build_keys = build_keys
                        .as_ref()
                        .ok_or_else(|| "hash join requires gathered build keys".to_owned())?;
                    let probe_keys = probe_keys
                        .as_ref()
                        .ok_or_else(|| "hash join requires gathered probe keys".to_owned())?;
                    let (ht_k, ht_i) = metal::hash_join_build_gv(build_keys, ht_capacity, config)?;
                    metal::hash_join_probe_gv(probe_keys, &ht_k, &ht_i, ht_capacity, config)?
                }
                GpuJoinStrategy::DenseLookup {
                    min_key,
                    slot_count,
                } => {
                    if let Some(build_keys) = build_keys.as_ref() {
                        let probe_keys = probe_keys
                            .as_ref()
                            .ok_or_else(|| "dense join requires gathered probe keys".to_owned())?;
                        let lookup_rows = metal::dense_lookup_build_i32_gv(
                            build_keys, min_key, slot_count, config,
                        )?;
                        metal::dense_lookup_probe_i32_gv(
                            probe_keys,
                            &lookup_rows,
                            min_key,
                            slot_count,
                            config,
                        )?
                    } else {
                        let lookup_rows = metal::dense_lookup_build_from_indices_i32_gv(
                            &build_col_gpu,
                            &gv_indices[build_table_idx],
                            min_key,
                            slot_count,
                            config,
                        )?;
                        metal::dense_lookup_probe_from_indices_i32_gv(
                            &probe_col_gpu,
                            &gv_indices[src_idx],
                            &lookup_rows,
                            min_key,
                            slot_count,
                            config,
                        )?
                    }
                }
            };

            let (probe_kept, build_kept) = metal::compact_matches_gv(&matched, config)?;
            let live_after = &live_after_steps[step_idx];
            let mut next_joined_sources = Vec::with_capacity(joined_sources.len() + 1);
            for &joined_idx in &joined_sources {
                if live_after[joined_idx] {
                    gv_indices[joined_idx] =
                        metal::reindex_gv(&gv_indices[joined_idx], &build_kept, config)?;
                    next_joined_sources.push(joined_idx);
                }
            }
            if live_after[src_idx] {
                gv_indices[src_idx] = metal::reindex_gv(&gv_indices[src_idx], &probe_kept, config)?;
                next_joined_sources.push(src_idx);
            }
            joined_sources = next_joined_sources;
        }
        false
    };
    profile_log(profile, "encode filter+join", encode_join_start);

    let join_wait_start = Instant::now();
    metal::end_batch(config)?;
    profile_log(profile, "execute filter+join", join_wait_start);

    let final_row_source_idx = if used_cached_join_rows || use_candidate_topk {
        representative_plan.table_idx
    } else {
        joined_sources.first().copied().unwrap_or(base_source_idx)
    };
    let final_join_count = gv_indices[final_row_source_idx].download_count();
    if final_join_count == 0 {
        let join_result = JoinResult {
            tables: plan
                .sources
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    (
                        s.alias.as_deref().unwrap_or(&s.name).to_owned(),
                        pinned_tables[i].clone(),
                    )
                })
                .collect(),
            table_indices: vec![Vec::new(); plan.sources.len()],
        };
        return execute_from_lookup(plan, &JoinResultLookup(&join_result), config);
    }
    for &joined_idx in &joined_sources {
        gv_indices[joined_idx].max_len = final_join_count;
    }
    if !used_cached_join_rows {
        if let (Some(cache_key), Some(candidate_topk)) =
            (join_output_cache_key.as_deref(), candidate_topk.as_ref())
        {
            metal::put_join_result_cache_gv(
                cache_key.to_owned(),
                &gv_indices[representative_plan.table_idx],
                &gv_indices[candidate_topk.aggregate_source_idx],
                config,
            );
        }
    }
    let candidate_representative_row_ids = candidate_topk
        .as_ref()
        .map(|_| gv_indices[representative_plan.table_idx].clone());

    let sum_expr = match grouped_agg_plan.aggregate_slots.first() {
        Some(AggregateSlotPlan {
            func: AggFunc::Sum,
            input: AggregateInputPlan::Expr(expr),
            ..
        }) => expr,
        _ => {
            metal::end_batch(config)?;
            return Ok(None);
        }
    };

    let fused_revenue_columns = fused_mul_one_minus_columns(sum_expr);
    let candidate_mode = requested_candidate_mode.or_else(|| {
        if candidate_topk.is_some() {
            Some(if fused_revenue_columns.is_some() {
                ExperimentalGroupedTopKMode::SlotFusedGpuTopK
            } else {
                ExperimentalGroupedTopKMode::CandidateCompact
            })
        } else {
            None
        }
    });
    if candidate_mode == Some(ExperimentalGroupedTopKMode::GenericGroupedAggregate) {
        let join_result =
            build_join_result_from_gpu_indices(plan, pinned_tables, &joined_sources, &gv_indices);
        return execute_from_lookup(plan, &JoinResultLookup(&join_result), config);
    }

    // ── Phase 3: Project -> Aggregate -> Top-K, still on device ───────
    let secondary_order_gpu = metal::column_to_gpu_buffer(secondary_order_col, config)?;

    let encode_agg_start = Instant::now();
    metal::begin_batch(config)?;
    let needs_materialized_sum = !matches!(
        candidate_mode,
        Some(ExperimentalGroupedTopKMode::SlotFusedGpuTopK)
            | Some(ExperimentalGroupedTopKMode::SlotFusedCpuTopK)
    );
    let sum_gv = if needs_materialized_sum {
        Some(
            if let Some((price_name, discount_name)) = fused_revenue_columns {
                let (price_tidx, price_col) = match resolve_column_gpu(price_name, &table_refs) {
                    Some(r) => r,
                    None => {
                        metal::end_batch(config)?;
                        return Ok(None);
                    }
                };
                let (discount_tidx, discount_col) =
                    match resolve_column_gpu(discount_name, &table_refs) {
                        Some(r) => r,
                        None => {
                            metal::end_batch(config)?;
                            return Ok(None);
                        }
                    };
                if price_tidx != discount_tidx {
                    metal::end_batch(config)?;
                    return Ok(None);
                }
                let price_gpu = metal::column_to_gpu_buffer(price_col, config)?;
                let discount_gpu = metal::column_to_gpu_buffer(discount_col, config)?;
                metal::project_revenue_from_indices_gv(
                    &price_gpu,
                    &discount_gpu,
                    &gv_indices[price_tidx],
                    config,
                )?
            } else {
                let Some(sum_gv) = evaluate_expr_gv(sum_expr, &table_refs, &gv_indices, config)?
                else {
                    metal::end_batch(config)?;
                    return Ok(None);
                };
                sum_gv
            },
        )
    } else {
        None
    };

    enum FastPathTopKOutput {
        Gpu {
            row_ids: GpuVector,
            aggregate_values: GpuVector,
        },
        Cpu {
            row_ids: Vec<u32>,
            aggregate_values: Vec<f32>,
        },
    }

    let topk_output = if let Some(candidate_topk) = &candidate_topk {
        let candidate_row_ids = candidate_representative_row_ids
            .as_ref()
            .expect("candidate grouped top-k requires representative candidates");
        let candidate_secondary_order_values =
            metal::gather_i32_gv(&secondary_order_gpu, candidate_row_ids, config)?;

        let representative_key_col = representative_table
            .column(&candidate_topk.representative_key_column)
            .ok_or_else(|| "candidate top-k representative key column missing".to_owned())?;
        let aggregate_key_col = pinned_tables[candidate_topk.aggregate_source_idx]
            .column(&candidate_topk.aggregate_key_column)
            .ok_or_else(|| "candidate top-k aggregate key column missing".to_owned())?;
        let representative_key_gpu = metal::column_to_gpu_buffer(representative_key_col, config)?;
        let aggregate_key_gpu = metal::column_to_gpu_buffer(aggregate_key_col, config)?;
        let join_strategy = choose_gpu_join_strategy(
            representative_table,
            &candidate_topk.representative_key_column,
            candidate_row_ids.max_len,
        )?;

        let matches = match join_strategy {
            GpuJoinStrategy::Hash { ht_capacity } => {
                let build_keys =
                    metal::gather_i32_gv(&representative_key_gpu, candidate_row_ids, config)?;
                let probe_keys = metal::gather_i32_gv(
                    &aggregate_key_gpu,
                    &gv_indices[candidate_topk.aggregate_source_idx],
                    config,
                )?;
                let (ht_k, ht_i) = metal::hash_join_build_gv(&build_keys, ht_capacity, config)?;
                metal::hash_join_probe_gv(&probe_keys, &ht_k, &ht_i, ht_capacity, config)?
            }
            GpuJoinStrategy::DenseLookup {
                min_key,
                slot_count,
            } => {
                let lookup_rows = metal::dense_lookup_build_from_indices_i32_gv(
                    &representative_key_gpu,
                    candidate_row_ids,
                    min_key,
                    slot_count,
                    config,
                )?;
                metal::dense_lookup_probe_from_indices_i32_gv(
                    &aggregate_key_gpu,
                    &gv_indices[candidate_topk.aggregate_source_idx],
                    &lookup_rows,
                    min_key,
                    slot_count,
                    config,
                )?
            }
        };
        match candidate_mode {
            Some(ExperimentalGroupedTopKMode::SlotValuesGpuTopK) => {
                let dense_sums = metal::accumulate_dense_slots_gv(
                    &matches,
                    sum_gv
                        .as_ref()
                        .expect("slot values path requires materialized aggregate input"),
                    candidate_row_ids,
                    config,
                )?;
                let dense_rows = metal::GpuDenseGroupedRows {
                    representative_row_ids: candidate_row_ids.clone(),
                    secondary_order_values: candidate_secondary_order_values,
                    aggregate_values: dense_sums,
                };
                let dense_topk = metal::topk_dense_grouped_rows(
                    &dense_rows,
                    grouped_topk.limit,
                    grouped_topk.secondary_order_group_idx.is_some(),
                    config,
                )?;
                FastPathTopKOutput::Gpu {
                    row_ids: dense_topk.representative_row_ids,
                    aggregate_values: dense_topk.aggregate_values,
                }
            }
            Some(ExperimentalGroupedTopKMode::SlotFusedGpuTopK)
            | Some(ExperimentalGroupedTopKMode::SlotFusedCpuTopK) => {
                let Some((price_name, discount_name)) = fused_revenue_columns else {
                    metal::end_batch(config)?;
                    return Ok(None);
                };
                let (price_tidx, price_col) = match resolve_column_gpu(price_name, &table_refs) {
                    Some(r) => r,
                    None => {
                        metal::end_batch(config)?;
                        return Ok(None);
                    }
                };
                let (discount_tidx, discount_col) =
                    match resolve_column_gpu(discount_name, &table_refs) {
                        Some(r) => r,
                        None => {
                            metal::end_batch(config)?;
                            return Ok(None);
                        }
                    };
                if price_tidx != discount_tidx || price_tidx != candidate_topk.aggregate_source_idx
                {
                    metal::end_batch(config)?;
                    return Ok(None);
                }
                let price_gpu = metal::column_to_gpu_buffer(price_col, config)?;
                let discount_gpu = metal::column_to_gpu_buffer(discount_col, config)?;
                let dense_sums = metal::accumulate_revenue_dense_slots_from_indices_gv(
                    &matches,
                    &price_gpu,
                    &discount_gpu,
                    &gv_indices[price_tidx],
                    candidate_row_ids,
                    config,
                )?;
                let use_secondary = grouped_topk.secondary_order_group_idx.is_some();
                if candidate_mode == Some(ExperimentalGroupedTopKMode::SlotFusedCpuTopK) {
                    profile_log(profile, "encode aggregate+topk", encode_agg_start);
                    let agg_wait_start = Instant::now();
                    metal::end_batch(config)?;
                    profile_log(profile, "execute aggregate+topk", agg_wait_start);

                    let dense_count = dense_sums.download_count();
                    let representative_row_ids: Vec<u32> =
                        candidate_row_ids.download_data_as(dense_count);
                    let secondary_values: Vec<i32> =
                        candidate_secondary_order_values.download_data_as(dense_count);
                    let aggregate_values: Vec<f32> = dense_sums.download_data_as(dense_count);

                    let mut ranked: Vec<(u32, f32, i32)> = representative_row_ids
                        .into_iter()
                        .zip(aggregate_values)
                        .zip(secondary_values)
                        .map(|((row_id, sum), secondary)| (row_id, sum, secondary))
                        .filter(|(_, sum, _)| *sum != 0.0)
                        .collect();
                    ranked.sort_by(|lhs, rhs| {
                        let lhs_better = grouped_candidate_better(
                            lhs.1,
                            lhs.2,
                            lhs.0,
                            rhs.1,
                            rhs.2,
                            rhs.0,
                            use_secondary,
                        );
                        let rhs_better = grouped_candidate_better(
                            rhs.1,
                            rhs.2,
                            rhs.0,
                            lhs.1,
                            lhs.2,
                            lhs.0,
                            use_secondary,
                        );
                        match (lhs_better, rhs_better) {
                            (true, false) => std::cmp::Ordering::Less,
                            (false, true) => std::cmp::Ordering::Greater,
                            _ => std::cmp::Ordering::Equal,
                        }
                    });
                    ranked.truncate(grouped_topk.limit);
                    FastPathTopKOutput::Cpu {
                        row_ids: ranked.iter().map(|(row_id, _, _)| *row_id).collect(),
                        aggregate_values: ranked.iter().map(|(_, sum, _)| *sum).collect(),
                    }
                } else {
                    let dense_rows = metal::GpuDenseGroupedRows {
                        representative_row_ids: candidate_row_ids.clone(),
                        secondary_order_values: candidate_secondary_order_values,
                        aggregate_values: dense_sums,
                    };
                    let dense_topk = metal::topk_dense_grouped_rows(
                        &dense_rows,
                        grouped_topk.limit,
                        use_secondary,
                        config,
                    )?;
                    FastPathTopKOutput::Gpu {
                        row_ids: dense_topk.representative_row_ids,
                        aggregate_values: dense_topk.aggregate_values,
                    }
                }
            }
            Some(ExperimentalGroupedTopKMode::CandidateCompact) | None => {
                let diag = profile_enabled();
                // [DIAG] flush candidate hash_join phase
                if diag {
                    let t = Instant::now();
                    metal::end_batch(config)?;
                    eprintln!("[DIAG B: candidate gather+hash_join]  {:?}", t.elapsed());
                    metal::begin_batch(config)?;
                }
                let (probe_kept, build_kept) = metal::compact_matches_gv(&matches, config)?;
                let candidate_sums = metal::reindex_gv(
                    sum_gv
                        .as_ref()
                        .expect("candidate compact path requires materialized aggregate input"),
                    &probe_kept,
                    config,
                )?;
                // [DIAG] flush compact+reindex
                if diag {
                    let t = Instant::now();
                    metal::end_batch(config)?;
                    eprintln!("[DIAG C: compact_matches+reindex]  {:?}", t.elapsed());
                    eprintln!(
                        "[DIAG   build_kept.max_len={} probe_kept.max_len={} matches.max_len={}",
                        build_kept.max_len, probe_kept.max_len, matches.max_len
                    );
                    metal::begin_batch(config)?;
                }
                let dense_topk = metal::aggregate_topk_grouped_rows_gv(
                    &build_kept,
                    &candidate_secondary_order_values.data,
                    &candidate_sums,
                    grouped_topk.limit,
                    grouped_topk.secondary_order_group_idx.is_some(),
                    config,
                )?;
                // [DIAG] flush agg_topk core
                if diag {
                    let t = Instant::now();
                    metal::end_batch(config)?;
                    eprintln!("[DIAG D: aggregate_topk_grouped_rows_gv]  {:?}", t.elapsed());
                    metal::begin_batch(config)?;
                }
                let actual_row_ids = metal::reindex_gv(
                    candidate_row_ids,
                    &dense_topk.representative_row_ids,
                    config,
                )?;
                FastPathTopKOutput::Gpu {
                    row_ids: actual_row_ids,
                    aggregate_values: dense_topk.aggregate_values,
                }
            }
            Some(ExperimentalGroupedTopKMode::GenericGroupedAggregate) => unreachable!(),
        }
    } else {
        let topk_rows = metal::aggregate_topk_grouped_rows_gv(
            &gv_indices[representative_plan.table_idx],
            &secondary_order_gpu,
            sum_gv
                .as_ref()
                .expect("default grouped top-k path requires materialized aggregate input"),
            grouped_topk.limit,
            grouped_topk.secondary_order_group_idx.is_some(),
            config,
        )?;
        FastPathTopKOutput::Gpu {
            row_ids: topk_rows.representative_row_ids,
            aggregate_values: topk_rows.aggregate_values,
        }
    };
    let (representative_row_ids, aggregate_vals) = match topk_output {
        FastPathTopKOutput::Gpu {
            row_ids,
            aggregate_values,
        } => {
            profile_log(profile, "encode aggregate+topk", encode_agg_start);
            let agg_wait_start = Instant::now();
            metal::end_batch(config)?;
            profile_log(profile, "execute aggregate+topk", agg_wait_start);
            let final_count = aggregate_values.download_count();
            (
                row_ids.download_data_as(final_count),
                aggregate_values.download_data_as(final_count),
            )
        }
        FastPathTopKOutput::Cpu {
            row_ids,
            aggregate_values,
        } => (row_ids, aggregate_values),
    };

    let materialize_start = Instant::now();
    let final_count = aggregate_vals.len();
    if final_count == 0 {
        let join_result = JoinResult {
            tables: plan
                .sources
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    (
                        s.alias.as_deref().unwrap_or(&s.name).to_owned(),
                        pinned_tables[i].clone(),
                    )
                })
                .collect(),
            table_indices: vec![Vec::new(); plan.sources.len()],
        };
        return execute_from_lookup(plan, &JoinResultLookup(&join_result), config);
    }
    if query_result_cache_enabled() {
        source_cache::insert_grouped_topk_result(
            grouped_topk_result_cache_key,
            &representative_row_ids,
            &aggregate_vals,
        );
    }
    let result = materialize_grouped_topk_result(
        plan,
        pinned_tables,
        &grouped_agg_plan,
        &representative_plan,
        &representative_row_ids,
        &aggregate_vals,
    )?;
    profile_log(profile, "download+materialize", materialize_start);

    Ok(Some(result))
}

fn execute_multi_table(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    source_selections: Vec<SourceSelection>,
    execution_order: &[usize],
    config: &BufferConfig,
) -> Result<Option<MaterializedQueryResult>, Box<dyn Error>> {
    let Some(join_steps) = join_execution_steps(plan, pinned_tables, execution_order) else {
        return Ok(None);
    };
    let live_after_steps =
        live_sources_after_join_steps(plan, pinned_tables, execution_order, &join_steps, &[]);

    // Late materialization: carry per-table index arrays through the join
    // loop instead of copying column data.
    let mut join_result = JoinResult {
        tables: Vec::new(),
        table_indices: Vec::new(),
    };
    let mut joined_source_ids = Vec::new();

    for (order_pos, &src_idx) in execution_order.iter().enumerate() {
        let src = &plan.sources[src_idx];
        let table = &pinned_tables[src_idx];
        let src_ref = src.alias.as_deref().unwrap_or(src.name.as_str());
        let indices = source_selections[src_idx].final_indices.clone();

        if order_pos == 0 {
            // First table: just store a reference and filtered indices.
            join_result
                .tables
                .push((src_ref.to_owned(), pinned_tables[src_idx].clone()));
            join_result.table_indices.push(indices);
            joined_source_ids.push(src_idx);
        } else {
            // Join with running result.
            let step = &join_steps[order_pos - 1];
            let build_col_name = &step.build_col_name;
            let probe_col_name = &step.probe_col_name;

            // Extract build-side join key from original pinned table via index array.
            let build_lookup = JoinResultLookup(&join_result);
            let (build_col, build_indices) = build_lookup
                .get_column(build_col_name.as_str())
                .ok_or_else(|| {
                    format!("join key {build_col_name:?} not found in running result")
                })?;
            let build_keys: Vec<i32> = match build_col {
                PinnedColumn::Int32(v) | PinnedColumn::Date32(v) => {
                    build_indices.iter().map(|&i| v[i]).collect()
                }
                PinnedColumn::Int64(v) => build_indices.iter().map(|&i| v[i] as i32).collect(),
                _ => return Ok(None),
            };

            // Extract probe-side join key from this table's filtered rows.
            let probe_col = table
                .column(probe_col_name.as_str())
                .ok_or_else(|| format!("pinned table missing join column {probe_col_name:?}"))?;
            let probe_keys: Vec<i32> = match probe_col {
                PinnedColumn::Int32(v) | PinnedColumn::Date32(v) => {
                    indices.iter().map(|&i| v[i]).collect()
                }
                PinnedColumn::Int64(v) => indices.iter().map(|&i| v[i] as i32).collect(),
                _ => return Ok(None),
            };

            // GPU hash join.
            let ht_capacity = (build_keys.len().saturating_mul(2).max(16)).next_power_of_two();
            let ht_capacity = u32::try_from(ht_capacity)
                .map_err(|_| "hash join capacity overflowed u32".to_owned())?;
            let matched =
                metal::hash_join_probe_i32(&build_keys, &probe_keys, ht_capacity, config)?;

            let mut probe_kept: Vec<usize> = Vec::new();
            let mut build_kept: Vec<usize> = Vec::new();
            for (probe_idx, &build_row_idx) in matched.iter().enumerate() {
                if build_row_idx >= 0 {
                    probe_kept.push(probe_idx);
                    build_kept.push(build_row_idx as usize);
                }
            }

            let live_after = &live_after_steps[order_pos - 1];
            let mut next_tables = Vec::with_capacity(join_result.tables.len() + 1);
            let mut next_table_indices = Vec::with_capacity(join_result.table_indices.len() + 1);
            let mut next_joined_source_ids = Vec::with_capacity(joined_source_ids.len() + 1);

            for (table_pos, &joined_source_idx) in joined_source_ids.iter().enumerate() {
                if live_after[joined_source_idx] {
                    next_tables.push(join_result.tables[table_pos].clone());
                    next_table_indices.push(
                        build_kept
                            .iter()
                            .map(|&bi| join_result.table_indices[table_pos][bi])
                            .collect(),
                    );
                    next_joined_source_ids.push(joined_source_idx);
                }
            }

            if live_after[src_idx] {
                let new_table_indices: Vec<usize> =
                    probe_kept.iter().map(|&pi| indices[pi]).collect();
                next_tables.push((src_ref.to_owned(), pinned_tables[src_idx].clone()));
                next_table_indices.push(new_table_indices);
                next_joined_source_ids.push(src_idx);
            }

            join_result.tables = next_tables;
            join_result.table_indices = next_table_indices;
            joined_source_ids = next_joined_source_ids;
        }

        if join_result.row_count() == 0 && order_pos > 0 {
            let lookup = JoinResultLookup(&join_result);
            let schema = build_output_schema(plan, &lookup)?;
            return Ok(Some(MaterializedQueryResult::from_batches(
                Arc::new(schema),
                Vec::new(),
            )));
        }
    }

    let lookup = JoinResultLookup(&join_result);
    execute_from_lookup(plan, &lookup, config)
}

fn find_join_for_source(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    probe_source_idx: usize,
    joined_sources: &[usize],
) -> Option<(usize, String, String)> {
    for join in &plan.joins {
        let (left_idx, left_col) =
            resolve_join_endpoint(&join.left_table, &join.left_col, plan, pinned_tables)?;
        let (right_idx, right_col) =
            resolve_join_endpoint(&join.right_table, &join.right_col, plan, pinned_tables)?;

        if left_idx == probe_source_idx && joined_sources.contains(&right_idx) {
            return Some((right_idx, right_col, left_col));
        }
        if right_idx == probe_source_idx && joined_sources.contains(&left_idx) {
            return Some((left_idx, left_col, right_col));
        }
    }
    None
}

fn is_grouped_topk_pilot(
    plan: &GpuPlan,
    pinned_tables: &[Arc<PinnedTable>],
    execution_order: &[usize],
) -> bool {
    if plan.sources.len() < 2 || plan.joins.is_empty() {
        return false;
    }

    let Some(topk) = grouped_topk_plan(plan) else {
        return false;
    };

    if plan.filters.iter().any(|filter| {
        let Some((table_idx, col)) = resolve_column_in_sources(filter, plan, pinned_tables) else {
            return true;
        };
        !pinned_tables[table_idx].is_pilot_supported_column(&filter.column)
            || !filter_runs_on_gpu(col, filter)
    }) {
        return false;
    }

    let Some(agg) = &plan.aggregate else {
        return false;
    };

    if agg.group_by.is_empty() {
        return false;
    }

    if agg.group_by.iter().any(|name| {
        let Some((table_idx, col)) = resolve_column_name_in_sources(name, plan, pinned_tables)
        else {
            return true;
        };
        !pinned_tables[table_idx].is_pilot_supported_column(column_name_only(name))
            || matches!(
                col,
                PinnedColumn::DictionaryI32 { .. } | PinnedColumn::Float64(_)
            )
    }) {
        return false;
    }

    if plan.projections.iter().any(|proj| match &proj.expr {
        GpuExpr::Column(name) => !agg
            .group_by
            .iter()
            .any(|group_col| group_col.eq_ignore_ascii_case(name)),
        GpuExpr::AggCall {
            func: AggFunc::Sum, ..
        } => false,
        _ => true,
    }) {
        return false;
    }

    let mut joined_sources = vec![execution_order[0]];
    for &src_idx in execution_order.iter().skip(1) {
        let Some((build_table_idx, build_col_name, _probe_col_name)) =
            find_join_for_source(plan, pinned_tables, src_idx, &joined_sources)
        else {
            return false;
        };
        if !pinned_tables[build_table_idx].is_known_unique(build_col_name.as_str()) {
            return false;
        }
        joined_sources.push(src_idx);
    }

    build_representative_materialize_plan(plan, pinned_tables, agg, &topk).is_some()
}

fn resolve_column_in_sources<'a>(
    filter: &GpuFilter,
    plan: &'a GpuPlan,
    pinned_tables: &'a [Arc<PinnedTable>],
) -> Option<(usize, &'a PinnedColumn)> {
    let target_col = filter.column.to_lowercase();
    for (idx, source) in plan.sources.iter().enumerate() {
        if let Some(table_ref) = filter.table.as_deref() {
            let source_ref = source.alias.as_deref().unwrap_or(source.name.as_str());
            if !refs_eq(table_ref, source_ref) && !refs_eq(table_ref, source.name.as_str()) {
                continue;
            }
        }
        if let Some(col) = pinned_tables[idx].column(&target_col) {
            return Some((idx, col));
        }
    }
    None
}

fn resolve_column_name_in_sources<'a>(
    name: &str,
    plan: &'a GpuPlan,
    pinned_tables: &'a [Arc<PinnedTable>],
) -> Option<(usize, &'a PinnedColumn)> {
    let lowered = name.to_lowercase();
    if let Some((table_name, col_name)) = lowered.split_once('.') {
        for (idx, source) in plan.sources.iter().enumerate() {
            let source_ref = source.alias.as_deref().unwrap_or(source.name.as_str());
            if refs_eq(table_name, source_ref) || refs_eq(table_name, source.name.as_str()) {
                if let Some(col) = pinned_tables[idx].column(col_name) {
                    return Some((idx, col));
                }
            }
        }
        return None;
    }
    for (idx, table) in pinned_tables.iter().enumerate() {
        if let Some(col) = table.column(&lowered) {
            return Some((idx, col));
        }
    }
    None
}

fn column_name_only(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

/// Compute filtered row indices for a table given a set of filters.
/// Uses GPU-native filter → mask → indices pipeline when all filters are on
/// GPU-supported types. Falls back to CPU for f64/Utf8 filters.
fn compute_filter_indices(
    table: &PinnedTable,
    filters: &[&GpuFilter],
    config: &BufferConfig,
) -> Result<Vec<usize>, Box<dyn Error>> {
    // Try GPU-native path: all filters must be on GPU-supported column types.
    if let Some(indices) = try_gpu_filter_indices(table, filters, config)? {
        return Ok(indices);
    }

    // CPU fallback: compute bool masks and convert to indices.
    let mut combined: Option<Vec<bool>> = None;
    for filter in filters {
        let col = table
            .column(&filter.column)
            .ok_or_else(|| format!("pinned table missing filter column {:?}", filter.column))?;
        let m = match apply_single_filter(col, filter, config)? {
            Some(m) => m,
            None => return Err(format!("unsupported filter on column {:?}", filter.column).into()),
        };
        combined = Some(match combined {
            None => m,
            Some(mut prev) => {
                for (a, b) in prev.iter_mut().zip(m.iter()) {
                    *a &= *b;
                }
                prev
            }
        });
    }
    Ok(match combined {
        Some(m) => (0..table.row_count).filter(|&i| m[i]).collect(),
        None => (0..table.row_count).collect(),
    })
}

/// GPU-native filter path: compare → mask → AND → filter_to_indices, all on GPU.
fn try_gpu_filter_indices(
    table: &PinnedTable,
    filters: &[&GpuFilter],
    config: &BufferConfig,
) -> Result<Option<Vec<usize>>, Box<dyn Error>> {
    use crate::metal::GpuBuffer;

    let mut gpu_mask: Option<GpuBuffer> = None;

    // Batch compare + and_masks into one command buffer when there are multiple filters.
    let batching = filters.len() > 1;
    if batching {
        metal::begin_batch(config)?;
    }

    for filter in filters {
        let col = match table.column(&filter.column) {
            Some(c) => c,
            None => {
                if batching {
                    metal::end_batch(config)?;
                }
                return Ok(None);
            }
        };

        let mask = match gpu_mask_for_filter(col, filter, config)? {
            Some(mask) => mask,
            None => {
                if batching {
                    metal::end_batch(config)?;
                }
                return Ok(None);
            }
        };

        gpu_mask = Some(match gpu_mask {
            None => mask,
            Some(prev) => metal::and_masks_gpu(&prev, &mask, config)?,
        });
    }

    if batching {
        metal::end_batch(config)?;
    }

    let gpu_mask = match gpu_mask {
        Some(m) => m,
        None => return Ok(None),
    };

    // Convert GPU mask to index array, then download to CPU.
    let gpu_indices = metal::filter_to_indices_gpu(&gpu_mask, config)?;
    let indices_u32: Vec<u32> = gpu_indices.download_as();
    Ok(Some(indices_u32.iter().map(|&i| i as usize).collect()))
}

fn filter_runs_on_gpu(col: &PinnedColumn, filter: &GpuFilter) -> bool {
    match col {
        PinnedColumn::Int32(_)
        | PinnedColumn::Date32(_)
        | PinnedColumn::Int64(_)
        | PinnedColumn::Float32(_) => !matches!(filter.literal, GpuLiteral::Str(_)),
        PinnedColumn::Float64(_) => false,
        PinnedColumn::DictionaryI32 { .. } => {
            matches!(filter.op, ScalarCompareOp::Eq) && matches!(filter.literal, GpuLiteral::Str(_))
        }
    }
}

fn gpu_mask_for_filter(
    col: &PinnedColumn,
    filter: &GpuFilter,
    config: &BufferConfig,
) -> Result<Option<metal::GpuBuffer>, Box<dyn Error>> {
    if !filter_runs_on_gpu(col, filter) {
        return Ok(None);
    }

    let mask = match col {
        PinnedColumn::Int32(values) => {
            let scalar = literal_as_i32(&filter.literal)?;
            metal::compare_i32_to_gpu(values, scalar, filter.op, config)?
        }
        PinnedColumn::Date32(values) => {
            let scalar = literal_as_date32(&filter.literal)?;
            metal::compare_date32_to_gpu(values, scalar, filter.op, config)?
        }
        PinnedColumn::Int64(values) => {
            let scalar = literal_as_i64(&filter.literal);
            metal::compare_i64_to_gpu(values, scalar, filter.op, config)?
        }
        PinnedColumn::Float32(values) => {
            let scalar = literal_as_f32(&filter.literal);
            metal::compare_f32_to_gpu(values, scalar, filter.op, config)?
        }
        PinnedColumn::DictionaryI32 { codes, .. } => {
            let scalar = dictionary_literal_code(col, &filter.literal)?;
            metal::compare_i32_to_gpu(codes, scalar, filter.op, config)?
        }
        PinnedColumn::Float64(_) => return Ok(None),
    };
    Ok(Some(mask))
}

fn dictionary_literal_code(col: &PinnedColumn, lit: &GpuLiteral) -> Result<i32, Box<dyn Error>> {
    match lit {
        GpuLiteral::Str(value) => Ok(col.dictionary_code(value).unwrap_or(-1)),
        _ => Err("dictionary filter requires a string literal".into()),
    }
}

fn apply_single_filter(
    col: &PinnedColumn,
    filter: &GpuFilter,
    config: &BufferConfig,
) -> Result<Option<Vec<bool>>, Box<dyn Error>> {
    match col {
        PinnedColumn::Int32(values) => {
            let scalar = literal_as_i32(&filter.literal)?;
            Ok(Some(metal::compare_i32_values(
                values, scalar, filter.op, config,
            )?))
        }
        PinnedColumn::Date32(values) => {
            let scalar = literal_as_date32(&filter.literal)?;
            Ok(Some(metal::compare_date32_values(
                values, scalar, filter.op, config,
            )?))
        }
        PinnedColumn::Int64(values) => {
            let scalar = literal_as_i64(&filter.literal);
            Ok(Some(metal::compare_i64_values(
                values, scalar, filter.op, config,
            )?))
        }
        PinnedColumn::Float32(values) => {
            let scalar = literal_as_f32(&filter.literal);
            Ok(Some(metal::compare_f32_values(
                values, scalar, filter.op, config,
            )?))
        }
        PinnedColumn::Float64(values) => {
            let scalar = literal_as_f64(&filter.literal);
            Ok(Some(
                values
                    .iter()
                    .map(|&v| match filter.op {
                        ScalarCompareOp::Eq => v == scalar,
                        ScalarCompareOp::Lt => v < scalar,
                        ScalarCompareOp::Le => v <= scalar,
                        ScalarCompareOp::Gt => v > scalar,
                        ScalarCompareOp::Ge => v >= scalar,
                    })
                    .collect(),
            ))
        }
        PinnedColumn::DictionaryI32 { codes, .. } => {
            let target = dictionary_literal_code(col, &filter.literal)?;
            match filter.op {
                ScalarCompareOp::Eq => Ok(Some(codes.iter().map(|&code| code == target).collect())),
                _ => Ok(None),
            }
        }
    }
}

fn sort_record_batch(
    batch: &RecordBatch,
    order_by: &[GpuOrderBy],
) -> Result<RecordBatch, Box<dyn Error>> {
    use std::cmp::Ordering;

    let num_rows = batch.num_rows();
    let mut indices: Vec<usize> = (0..num_rows).collect();

    // Resolve sort column indices once.
    let sort_col_indices: Vec<(usize, bool)> = order_by
        .iter()
        .filter_map(|ob| {
            let idx = batch.schema().index_of(&ob.column).ok()?;
            Some((idx, ob.descending))
        })
        .collect();

    if sort_col_indices.is_empty() {
        return Ok(batch.clone());
    }

    // Use typed comparison to avoid f64 precision loss on i64 values.
    indices.sort_by(|&a, &b| {
        for &(col_idx, desc) in &sort_col_indices {
            let col = batch.column(col_idx);
            let cmp = compare_array_values(col, a, b);
            let cmp = if desc { cmp.reverse() } else { cmp };
            if cmp != Ordering::Equal {
                return cmp;
            }
        }
        Ordering::Equal
    });

    let new_columns: Vec<ArrayRef> = (0..batch.num_columns())
        .map(|col_idx| reorder_column(batch.column(col_idx), &indices))
        .collect();

    Ok(RecordBatch::try_new(batch.schema(), new_columns)?)
}

fn compare_array_values(col: &ArrayRef, a: usize, b: usize) -> std::cmp::Ordering {
    if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
        return arr.value(a).cmp(&arr.value(b));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        return arr.value(a).cmp(&arr.value(b));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
        return arr
            .value(a)
            .partial_cmp(&arr.value(b))
            .unwrap_or(std::cmp::Ordering::Equal);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        return arr
            .value(a)
            .partial_cmp(&arr.value(b))
            .unwrap_or(std::cmp::Ordering::Equal);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Date32Array>() {
        return arr.value(a).cmp(&arr.value(b));
    }
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        return arr.value(a).cmp(arr.value(b));
    }
    std::cmp::Ordering::Equal
}

fn reorder_column(col: &ArrayRef, indices: &[usize]) -> ArrayRef {
    if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
        return Arc::new(Int32Array::from(
            indices.iter().map(|&i| arr.value(i)).collect::<Vec<_>>(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        return Arc::new(Int64Array::from(
            indices.iter().map(|&i| arr.value(i)).collect::<Vec<_>>(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
        return Arc::new(Float32Array::from(
            indices.iter().map(|&i| arr.value(i)).collect::<Vec<_>>(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        return Arc::new(Float64Array::from(
            indices.iter().map(|&i| arr.value(i)).collect::<Vec<_>>(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<Date32Array>() {
        return Arc::new(Date32Array::from(
            indices.iter().map(|&i| arr.value(i)).collect::<Vec<_>>(),
        ));
    }
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        return Arc::new(StringArray::from(
            indices.iter().map(|&i| arr.value(i)).collect::<Vec<_>>(),
        ));
    }
    col.clone()
}

/// Abstraction for looking up pinned columns by name. Each column comes
/// with its own index array for late materialization — the caller gathers
/// from the original pinned data using these indices.
trait ColumnLookup {
    fn get_column(&self, name: &str) -> Option<(&PinnedColumn, &[usize])>;
    fn row_count(&self) -> usize;
    fn gpu_relation_bindings(&self) -> Option<(&[(String, Arc<PinnedTable>)], &[Vec<usize>])> {
        None
    }
}

/// Late-materialized join result: references to original pinned tables +
/// per-table row index arrays. All index arrays have the same length
/// (the join result cardinality). Columns are only gathered at the point
/// of consumption (GPU kernels, output construction).
struct JoinResult {
    /// (alias, original pinned table reference) per source table.
    tables: Vec<(String, Arc<PinnedTable>)>,
    /// Per-table row index arrays. `table_indices[t][i]` is the row index
    /// into `tables[t].1` for result row `i`. All arrays have the same length.
    table_indices: Vec<Vec<usize>>,
}

impl JoinResult {
    fn row_count(&self) -> usize {
        self.table_indices.first().map(|v| v.len()).unwrap_or(0)
    }

    fn resolve_column(&self, name: &str) -> Option<(usize, &PinnedColumn)> {
        resolve_column_gpu(name, &self.tables)
    }
}

struct JoinResultLookup<'a>(&'a JoinResult);

impl ColumnLookup for JoinResultLookup<'_> {
    fn get_column(&self, name: &str) -> Option<(&PinnedColumn, &[usize])> {
        let (table_idx, col) = self.0.resolve_column(name)?;
        Some((col, &self.0.table_indices[table_idx]))
    }
    fn row_count(&self) -> usize {
        self.0.row_count()
    }
    fn gpu_relation_bindings(&self) -> Option<(&[(String, Arc<PinnedTable>)], &[Vec<usize>])> {
        Some((&self.0.tables, &self.0.table_indices))
    }
}

/// Recursively evaluate a GpuExpr against pinned columns, producing f64 values.
/// Column indices come from the lookup (late materialization).
fn evaluate_expr(
    expr: &GpuExpr,
    lookup: &dyn ColumnLookup,
    config: &BufferConfig,
) -> Result<Vec<f64>, Box<dyn Error>> {
    match expr {
        GpuExpr::Column(col_name) => {
            let (col, indices) = lookup
                .get_column(col_name)
                .ok_or_else(|| format!("pinned table missing column {col_name:?}"))?;
            Ok(gather_column_as_f64(col, indices))
        }
        GpuExpr::Literal(lit) => {
            let val = literal_as_f64(lit);
            Ok(vec![val; lookup.row_count()])
        }
        GpuExpr::BinaryOp { op, left, right } => {
            if let Some(result) = try_scalar_projection(op, left, right, lookup, config)? {
                return Ok(result);
            }
            let left_vals = evaluate_expr(left, lookup, config)?;
            let right_vals = evaluate_expr(right, lookup, config)?;
            let metal_op = match op {
                GpuBinOp::Add => ScalarProjectOp::Add,
                GpuBinOp::Mul => ScalarProjectOp::Multiply,
                GpuBinOp::Sub => ScalarProjectOp::Subtract,
            };
            Ok(metal::elementwise_f64(
                &left_vals,
                &right_vals,
                metal_op,
                config,
            )?)
        }
        GpuExpr::AggCall { .. } => {
            Err("cannot evaluate aggregate call as a scalar expression".into())
        }
    }
}

fn gather_column_as_f64(col: &PinnedColumn, indices: &[usize]) -> Vec<f64> {
    match col {
        PinnedColumn::Int32(v) | PinnedColumn::Date32(v) => {
            indices.iter().map(|&i| v[i] as f64).collect()
        }
        PinnedColumn::Int64(v) => indices.iter().map(|&i| v[i] as f64).collect(),
        PinnedColumn::Float32(v) => indices.iter().map(|&i| v[i] as f64).collect(),
        PinnedColumn::Float64(v) => indices.iter().map(|&i| v[i]).collect(),
        PinnedColumn::DictionaryI32 { .. } => indices.iter().map(|_| 0.0).collect(),
    }
}

/// Fast path for BinaryOp(op, Column, Literal) or BinaryOp(op, Literal, Column).
/// Uses the scalar projection kernel directly (native f32, no literal broadcast,
/// no f64→f32→f64 round-trip for the literal side).
fn try_scalar_projection(
    op: &GpuBinOp,
    left: &GpuExpr,
    right: &GpuExpr,
    lookup: &dyn ColumnLookup,
    config: &BufferConfig,
) -> Result<Option<Vec<f64>>, Box<dyn Error>> {
    let (col_name, scalar_val, scalar_on_left) = match (left, right) {
        (GpuExpr::Column(c), GpuExpr::Literal(lit)) => (c.as_str(), literal_as_f64(lit), false),
        (GpuExpr::Literal(lit), GpuExpr::Column(c)) => (c.as_str(), literal_as_f64(lit), true),
        _ => return Ok(None),
    };

    let (col, indices) = match lookup.get_column(col_name) {
        Some(pair) => pair,
        None => return Ok(None),
    };

    let metal_op = match (op, scalar_on_left) {
        (GpuBinOp::Add, _) => ScalarProjectOp::Add,
        (GpuBinOp::Mul, _) => ScalarProjectOp::Multiply,
        (GpuBinOp::Sub, false) => ScalarProjectOp::Subtract,
        (GpuBinOp::Sub, true) => ScalarProjectOp::SubtractFrom,
    };

    let values_f32: Vec<f32> = match col {
        PinnedColumn::Int32(v) | PinnedColumn::Date32(v) => {
            indices.iter().map(|&i| v[i] as f32).collect()
        }
        PinnedColumn::Int64(v) => indices.iter().map(|&i| v[i] as f32).collect(),
        PinnedColumn::Float32(v) => indices.iter().map(|&i| v[i]).collect(),
        PinnedColumn::Float64(v) => indices.iter().map(|&i| v[i] as f32).collect(),
        PinnedColumn::DictionaryI32 { .. } => return Ok(None),
    };

    let result_f32 = metal::project_f32_values(&values_f32, scalar_val as f32, metal_op, config)?;
    Ok(Some(result_f32.iter().map(|&v| v as f64).collect()))
}

fn execute_aggregate_impl(
    plan: &GpuPlan,
    lookup: &dyn ColumnLookup,
    agg: &GpuAggregate,
    config: &BufferConfig,
) -> Result<Option<MaterializedQueryResult>, Box<dyn Error>> {
    if agg.group_by.is_empty() {
        return execute_ungrouped_aggregate(plan, lookup, config);
    }
    let Some(grouped_plan) = build_grouped_aggregate_plan(plan, agg) else {
        return Ok(None);
    };

    // GPU fast path: single integer group-by key with only SUM aggregates.
    if let Some(result) = try_gpu_grouped_aggregate(plan, lookup, &grouped_plan, config)? {
        return Ok(Some(result));
    }

    // CPU fallback: multi-key or non-SUM aggregates.
    let num_rows = lookup.row_count();
    if num_rows == 0 {
        let schema = build_grouped_aggregate_schema(&grouped_plan, lookup);
        return Ok(Some(MaterializedQueryResult::from_batches(
            Arc::new(schema),
            Vec::new(),
        )));
    }

    // Gather group-by columns with their per-table indices.
    let group_cols: Vec<(&PinnedColumn, &[usize])> = grouped_plan
        .group_by
        .iter()
        .map(|name| {
            lookup
                .get_column(name)
                .ok_or_else(|| format!("pinned table missing group-by column {name:?}").into())
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    // Build composite key → group index map.
    let mut key_to_group: HashMap<Vec<GroupKey>, usize> = HashMap::new();
    let mut group_positions: Vec<Vec<usize>> = Vec::new();

    for pos in 0..num_rows {
        let key: Vec<GroupKey> = group_cols
            .iter()
            .map(|(col, indices)| GroupKey::from_column(col, indices[pos]))
            .collect();
        let group_idx = match key_to_group.get(&key) {
            Some(&idx) => idx,
            None => {
                let idx = group_positions.len();
                key_to_group.insert(key, idx);
                group_positions.push(Vec::new());
                idx
            }
        };
        group_positions[group_idx].push(pos);
    }

    let num_groups = group_positions.len();

    // Pre-evaluate all aggregate inner expressions ONCE over all selected rows.
    // This avoids per-group GPU dispatches — the key performance optimization.
    let mut precomputed: Vec<Option<Vec<f64>>> = Vec::new();
    for slot in &grouped_plan.aggregate_slots {
        match &slot.input {
            AggregateInputPlan::CountStar => precomputed.push(None),
            AggregateInputPlan::Expr(expr) => {
                let values = evaluate_expr(expr, lookup, config)?;
                precomputed.push(Some(values));
            }
        }
    }

    // Build output columns.
    let mut fields = Vec::new();
    let mut output_columns: Vec<ArrayRef> = Vec::new();

    for output in &grouped_plan.outputs {
        match output {
            GroupedAggregateOutput::GroupKey { group_idx, alias } => {
                let col_name = &grouped_plan.group_by[*group_idx];
                let (col, col_indices) = lookup
                    .get_column(col_name)
                    .ok_or_else(|| format!("pinned table missing column {col_name:?}"))?;
                let representative_indices: Vec<usize> =
                    group_positions.iter().map(|g| col_indices[g[0]]).collect();
                fields.push(Field::new(alias, column_arrow_type(col), false));
                output_columns.push(gather_column(col, &representative_indices));
            }
            GroupedAggregateOutput::Aggregate { agg_idx } => {
                let slot = &grouped_plan.aggregate_slots[*agg_idx];
                let vals = &precomputed[*agg_idx];
                let mut group_results = Vec::with_capacity(num_groups);
                for group_pos in &group_positions {
                    let result = match slot.state_kind {
                        AggregateStateKind::CountU64 => group_pos.len() as f64,
                        AggregateStateKind::SumF64 => {
                            let v = vals.as_ref().unwrap();
                            group_pos.iter().map(|&p| v[p]).sum()
                        }
                        AggregateStateKind::AvgSumCount => {
                            let v = vals.as_ref().unwrap();
                            let sum: f64 = group_pos.iter().map(|&p| v[p]).sum();
                            sum / group_pos.len() as f64
                        }
                        AggregateStateKind::MinF64 => {
                            let v = vals.as_ref().unwrap();
                            group_pos
                                .iter()
                                .map(|&p| v[p])
                                .fold(f64::INFINITY, f64::min)
                        }
                        AggregateStateKind::MaxF64 => {
                            let v = vals.as_ref().unwrap();
                            group_pos
                                .iter()
                                .map(|&p| v[p])
                                .fold(f64::NEG_INFINITY, f64::max)
                        }
                    };
                    group_results.push(result);
                }
                fields.push(Field::new(&slot.alias, DataType::Float64, false));
                output_columns.push(Arc::new(Float64Array::from(group_results)));
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let mut batch = RecordBatch::try_new(schema.clone(), output_columns)?;

    if !plan.order_by.is_empty() {
        batch = sort_record_batch(&batch, &plan.order_by)?;
    }
    if let Some(limit) = plan.limit {
        if batch.num_rows() > limit {
            batch = batch.slice(0, limit);
        }
    }

    Ok(Some(MaterializedQueryResult::from_batches(
        batch.schema(),
        vec![batch],
    )))
}

fn upload_indices_as_gv(
    table_indices: &[Vec<usize>],
    config: &BufferConfig,
) -> Result<Vec<GpuVector>, Box<dyn Error>> {
    table_indices
        .iter()
        .map(|indices| {
            let gpu_indices: Vec<u32> = indices
                .iter()
                .map(|&idx| u32::try_from(idx).map_err(|_| "row index overflowed u32".to_owned()))
                .collect::<Result<_, _>>()?;
            metal::upload_gv(&gpu_indices, config).map_err(Box::<dyn Error>::from)
        })
        .collect()
}

fn build_group_key_array_from_i32_values(col: &PinnedColumn, keys: &[i32]) -> Option<ArrayRef> {
    match col {
        PinnedColumn::Int32(_) => Some(Arc::new(Int32Array::from(keys.to_vec()))),
        PinnedColumn::Date32(_) => Some(Arc::new(Date32Array::from(keys.to_vec()))),
        PinnedColumn::DictionaryI32 { values, .. } => {
            let decoded: Vec<String> = keys
                .iter()
                .map(|&code| {
                    usize::try_from(code)
                        .ok()
                        .and_then(|idx| values.get(idx))
                        .cloned()
                        .unwrap_or_default()
                })
                .collect();
            Some(Arc::new(StringArray::from(decoded)))
        }
        PinnedColumn::Int64(_) | PinnedColumn::Float32(_) | PinnedColumn::Float64(_) => None,
    }
}

fn gpu_state_kind(kind: AggregateStateKind) -> GpuGroupedAggregateStateKind {
    match kind {
        AggregateStateKind::CountU64 => GpuGroupedAggregateStateKind::Count,
        AggregateStateKind::SumF64 => GpuGroupedAggregateStateKind::Sum,
        AggregateStateKind::MinF64 => GpuGroupedAggregateStateKind::Min,
        AggregateStateKind::MaxF64 => GpuGroupedAggregateStateKind::Max,
        AggregateStateKind::AvgSumCount => GpuGroupedAggregateStateKind::Avg,
    }
}

/// GPU fast path for grouped aggregates: exact i32-like grouping with multi-state aggregates.
fn try_gpu_grouped_aggregate(
    plan: &GpuPlan,
    lookup: &dyn ColumnLookup,
    grouped_plan: &GroupedAggregatePlan,
    config: &BufferConfig,
) -> Result<Option<MaterializedQueryResult>, Box<dyn Error>> {
    let num_rows = lookup.row_count();
    if grouped_plan.group_by.is_empty()
        || grouped_plan.group_by.len() > 4
        || grouped_plan.aggregate_slots.is_empty()
    {
        return Ok(None);
    }

    if num_rows == 0 {
        let schema = build_grouped_aggregate_schema(grouped_plan, lookup);
        return Ok(Some(MaterializedQueryResult::from_batches(
            Arc::new(schema),
            Vec::new(),
        )));
    }

    let Some((table_refs, table_indices)) = lookup.gpu_relation_bindings() else {
        return Ok(None);
    };
    let gv_indices = upload_indices_as_gv(table_indices, config)?;

    let mut group_key_vectors = Vec::with_capacity(grouped_plan.group_by.len());
    let mut group_key_columns = Vec::with_capacity(grouped_plan.group_by.len());
    for group_name in &grouped_plan.group_by {
        let (table_idx, group_col) = match resolve_column_gpu(group_name, table_refs) {
            Some(pair) => pair,
            None => return Ok(None),
        };
        let Some(group_gv) = gather_group_key_as_i32_gv(group_col, &gv_indices[table_idx], config)?
        else {
            return Ok(None);
        };
        group_key_vectors.push(group_gv);
        group_key_columns.push(group_col);
    }

    let mut aggregate_input_vectors: Vec<Option<GpuVector>> =
        Vec::with_capacity(grouped_plan.aggregate_slots.len());
    for slot in &grouped_plan.aggregate_slots {
        match &slot.input {
            AggregateInputPlan::CountStar => aggregate_input_vectors.push(None),
            AggregateInputPlan::Expr(inner_expr) => {
                let Some(values_gv) =
                    evaluate_expr_gv(inner_expr, table_refs, &gv_indices, config)?
                else {
                    return Ok(None);
                };
                aggregate_input_vectors.push(Some(values_gv));
            }
        }
    }

    let aggregate_specs: Vec<GpuGroupedAggregateSpec<'_>> = grouped_plan
        .aggregate_slots
        .iter()
        .zip(aggregate_input_vectors.iter())
        .map(|(slot, values)| GpuGroupedAggregateSpec {
            state_kind: gpu_state_kind(slot.state_kind),
            values: values.as_ref(),
        })
        .collect();
    let gpu_rows = metal::grouped_aggregate_exact_gv(&group_key_vectors, &aggregate_specs, config)?;

    // Build output columns.
    let mut fields = Vec::new();
    let mut output_columns: Vec<ArrayRef> = Vec::new();

    for output in &grouped_plan.outputs {
        match output {
            GroupedAggregateOutput::GroupKey { group_idx, alias } => {
                let col = group_key_columns[*group_idx];
                let Some(array) =
                    build_group_key_array_from_i32_values(col, &gpu_rows.key_lanes[*group_idx])
                else {
                    return Ok(None);
                };
                fields.push(Field::new(alias, column_arrow_type(col), false));
                output_columns.push(array);
            }
            GroupedAggregateOutput::Aggregate { agg_idx } => {
                let slot = &grouped_plan.aggregate_slots[*agg_idx];
                let gpu_res = &gpu_rows.slots[*agg_idx];
                let values_f64: Vec<f64> = match slot.state_kind {
                    AggregateStateKind::CountU64 => gpu_res
                        .counts
                        .as_ref()
                        .ok_or("gpu grouped aggregate count slot is missing counts")?
                        .iter()
                        .map(|&count| count as f64)
                        .collect(),
                    AggregateStateKind::SumF64
                    | AggregateStateKind::MinF64
                    | AggregateStateKind::MaxF64 => gpu_res
                        .values
                        .as_ref()
                        .ok_or("gpu grouped aggregate value slot is missing values")?
                        .iter()
                        .map(|&value| value as f64)
                        .collect(),
                    AggregateStateKind::AvgSumCount => {
                        let sums = gpu_res
                            .values
                            .as_ref()
                            .ok_or("gpu grouped aggregate avg slot is missing sums")?;
                        let counts = gpu_res
                            .counts
                            .as_ref()
                            .ok_or("gpu grouped aggregate avg slot is missing counts")?;
                        sums.iter()
                            .zip(counts.iter())
                            .map(|(&sum, &count)| {
                                if count == 0 {
                                    0.0
                                } else {
                                    f64::from(sum) / count as f64
                                }
                            })
                            .collect()
                    }
                };
                fields.push(Field::new(&slot.alias, DataType::Float64, false));
                output_columns.push(Arc::new(Float64Array::from(values_f64)));
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let mut batch = RecordBatch::try_new(schema.clone(), output_columns)?;

    if !plan.order_by.is_empty() {
        batch = sort_record_batch(&batch, &plan.order_by)?;
    }
    if let Some(limit) = plan.limit {
        if batch.num_rows() > limit {
            batch = batch.slice(0, limit);
        }
    }

    Ok(Some(MaterializedQueryResult::from_batches(
        batch.schema(),
        vec![batch],
    )))
}

fn execute_ungrouped_aggregate(
    plan: &GpuPlan,
    lookup: &dyn ColumnLookup,
    config: &BufferConfig,
) -> Result<Option<MaterializedQueryResult>, Box<dyn Error>> {
    let mut fields = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();

    for proj in &plan.projections {
        match &proj.expr {
            GpuExpr::AggCall { func, arg } => {
                let value = compute_ungrouped_agg(*func, arg, lookup, config)?;
                match value {
                    Some(v) => {
                        fields.push(Field::new(&proj.alias, DataType::Float64, false));
                        columns.push(Arc::new(Float64Array::from(vec![v])));
                    }
                    None => return Ok(None),
                }
            }
            GpuExpr::Column(_) => return Ok(None),
            _ => return Ok(None),
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    Ok(Some(MaterializedQueryResult::from_batches(
        schema,
        vec![batch],
    )))
}

fn build_grouped_aggregate_schema(
    grouped_plan: &GroupedAggregatePlan,
    lookup: &dyn ColumnLookup,
) -> Schema {
    let fields: Vec<Field> = grouped_plan
        .outputs
        .iter()
        .map(|output| match output {
            GroupedAggregateOutput::GroupKey { alias, .. } => Field::new(
                alias,
                grouped_aggregate_output_type(output, grouped_plan, lookup),
                false,
            ),
            GroupedAggregateOutput::Aggregate { agg_idx } => Field::new(
                &grouped_plan.aggregate_slots[*agg_idx].alias,
                grouped_aggregate_output_type(output, grouped_plan, lookup),
                false,
            ),
        })
        .collect();
    Schema::new(fields)
}

/// Hashable group key value for CPU-side grouped aggregation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum GroupKey {
    Int32(i32),
    Int64(i64),
    Date32(i32),
    Str(String),
    /// For f32/f64 we use the bit pattern for hashing.
    F64Bits(u64),
}

impl GroupKey {
    fn from_column(col: &PinnedColumn, idx: usize) -> Self {
        match col {
            PinnedColumn::Int32(v) => GroupKey::Int32(v[idx]),
            PinnedColumn::Int64(v) => GroupKey::Int64(v[idx]),
            PinnedColumn::Date32(v) => GroupKey::Date32(v[idx]),
            PinnedColumn::Float32(v) => GroupKey::F64Bits((v[idx] as f64).to_bits()),
            PinnedColumn::Float64(v) => GroupKey::F64Bits(v[idx].to_bits()),
            PinnedColumn::DictionaryI32 { codes, values, .. } => {
                let value = usize::try_from(codes[idx])
                    .ok()
                    .and_then(|code| values.get(code))
                    .cloned()
                    .unwrap_or_default();
                GroupKey::Str(value)
            }
        }
    }
}

fn compute_ungrouped_agg(
    func: AggFunc,
    arg: &GpuExpr,
    lookup: &dyn ColumnLookup,
    config: &BufferConfig,
) -> Result<Option<f64>, Box<dyn Error>> {
    let num_rows = lookup.row_count();
    if func == AggFunc::Count && matches!(arg, GpuExpr::Column(c) if c == "*") {
        return Ok(Some(num_rows as f64));
    }

    if num_rows == 0 {
        return Ok(Some(0.0));
    }

    let values = evaluate_expr(arg, lookup, config)?;

    let result = match func {
        AggFunc::Count => values.len() as f64,
        AggFunc::Sum => values.iter().sum(),
        AggFunc::Avg => {
            let sum: f64 = values.iter().sum();
            sum / values.len() as f64
        }
        AggFunc::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
        AggFunc::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    };

    Ok(Some(result))
}

fn build_output_schema(
    plan: &GpuPlan,
    lookup: &dyn ColumnLookup,
) -> Result<Schema, Box<dyn Error>> {
    let mut fields = Vec::new();
    for proj in &plan.projections {
        let data_type = match &proj.expr {
            GpuExpr::Column(col_name) => {
                let (col, _) = lookup
                    .get_column(col_name)
                    .ok_or_else(|| format!("pinned table missing column {col_name:?}"))?;
                column_arrow_type(col)
            }
            GpuExpr::BinaryOp { .. } | GpuExpr::Literal(_) => DataType::Float64,
            GpuExpr::AggCall { .. } => DataType::Float64,
        };
        fields.push(Field::new(&proj.alias, data_type, false));
    }
    Ok(Schema::new(fields))
}

fn column_arrow_type(col: &PinnedColumn) -> DataType {
    match col {
        PinnedColumn::Int32(_) => DataType::Int32,
        PinnedColumn::Int64(_) => DataType::Int64,
        PinnedColumn::Float32(_) => DataType::Float32,
        PinnedColumn::Float64(_) => DataType::Float64,
        PinnedColumn::Date32(_) => DataType::Date32,
        PinnedColumn::DictionaryI32 { .. } => DataType::Utf8,
    }
}

fn gather_column(col: &PinnedColumn, indices: &[usize]) -> ArrayRef {
    match col {
        PinnedColumn::Int32(v) => Arc::new(Int32Array::from(
            indices.iter().map(|&i| v[i]).collect::<Vec<_>>(),
        )),
        PinnedColumn::Int64(v) => Arc::new(Int64Array::from(
            indices.iter().map(|&i| v[i]).collect::<Vec<_>>(),
        )),
        PinnedColumn::Float32(v) => Arc::new(Float32Array::from(
            indices.iter().map(|&i| v[i]).collect::<Vec<_>>(),
        )),
        PinnedColumn::Float64(v) => Arc::new(Float64Array::from(
            indices.iter().map(|&i| v[i]).collect::<Vec<_>>(),
        )),
        PinnedColumn::Date32(v) => Arc::new(Date32Array::from(
            indices.iter().map(|&i| v[i]).collect::<Vec<_>>(),
        )),
        PinnedColumn::DictionaryI32 { codes, values, .. } => {
            let arr: Vec<&str> = indices
                .iter()
                .filter_map(|&i| {
                    usize::try_from(codes[i])
                        .ok()
                        .and_then(|code| values.get(code))
                        .map(String::as_str)
                })
                .collect();
            Arc::new(duckdb::arrow::array::StringArray::from(arr))
        }
    }
}

fn literal_as_i32(lit: &GpuLiteral) -> Result<i32, Box<dyn Error>> {
    match lit {
        GpuLiteral::Int(i) => i32::try_from(*i).map_err(|_| "literal overflows i32".into()),
        GpuLiteral::Float(f) => Ok(*f as i32),
        GpuLiteral::Date(d) => Ok(*d),
        GpuLiteral::Str(_) => Err("cannot convert string to i32".into()),
    }
}

fn literal_as_date32(lit: &GpuLiteral) -> Result<i32, Box<dyn Error>> {
    match lit {
        GpuLiteral::Date(d) => Ok(*d),
        GpuLiteral::Int(i) => i32::try_from(*i).map_err(|_| "literal overflows i32".into()),
        _ => Err("expected date or integer literal for date comparison".into()),
    }
}

fn literal_as_f64(lit: &GpuLiteral) -> f64 {
    match lit {
        GpuLiteral::Int(i) => *i as f64,
        GpuLiteral::Float(f) => *f,
        GpuLiteral::Date(d) => *d as f64,
        GpuLiteral::Str(_) => 0.0,
    }
}

fn literal_as_i64(lit: &GpuLiteral) -> i64 {
    match lit {
        GpuLiteral::Int(i) => *i,
        GpuLiteral::Float(f) => *f as i64,
        GpuLiteral::Date(d) => *d as i64,
        GpuLiteral::Str(_) => 0,
    }
}

fn literal_as_f32(lit: &GpuLiteral) -> f32 {
    match lit {
        GpuLiteral::Int(i) => *i as f32,
        GpuLiteral::Float(f) => *f as f32,
        GpuLiteral::Date(d) => *d as f32,
        GpuLiteral::Str(_) => 0.0,
    }
}

fn flatten_object_name(name: &ObjectName) -> String {
    name.0
        .last()
        .map(|part| {
            part.as_ident()
                .map(|id| id.value.clone())
                .unwrap_or_default()
        })
        .unwrap_or_default()
}

fn parse_table_factor(factor: &ast::TableFactor) -> Option<TableSource> {
    match factor {
        ast::TableFactor::Table { name, alias, .. } => Some(TableSource {
            name: flatten_object_name(name),
            alias: alias.as_ref().map(|a| a.name.value.clone()),
        }),
        _ => None,
    }
}

fn parse_join_constraint(
    join_op: &ast::JoinOperator,
    existing_sources: &[TableSource],
    right_source: &TableSource,
) -> Option<GpuJoin> {
    let constraint = match join_op {
        ast::JoinOperator::Join(c)
        | ast::JoinOperator::Inner(c)
        | ast::JoinOperator::LeftOuter(c)
        | ast::JoinOperator::RightOuter(c)
        | ast::JoinOperator::FullOuter(c, ..) => c,
        ast::JoinOperator::CrossJoin => return None,
        _ => return None,
    };
    match constraint {
        ast::JoinConstraint::On(expr) => parse_equi_join_expr(expr, existing_sources, right_source),
        _ => None,
    }
}

fn parse_equi_join_expr(
    expr: &Expr,
    existing_sources: &[TableSource],
    right_source: &TableSource,
) -> Option<GpuJoin> {
    // Handle: left_expr = right_expr
    if let Expr::BinaryOp {
        left,
        op: BinaryOperator::Eq,
        right,
    } = expr
    {
        let left_qc = extract_qualified_column(left)?;
        let right_qc = extract_qualified_column(right)?;

        let right_name = right_source
            .alias
            .as_deref()
            .unwrap_or(right_source.name.as_str());

        // If both columns are qualified, use the table refs directly.
        if let (Some(left_ref), Some(right_ref)) =
            (left_qc.table.as_deref(), right_qc.table.as_deref())
        {
            if source_matches(existing_sources, left_ref) && refs_eq(right_ref, right_name) {
                return Some(GpuJoin {
                    left_table: left_ref.to_owned(),
                    left_col: left_qc.column,
                    right_table: right_ref.to_owned(),
                    right_col: right_qc.column,
                });
            }
            if source_matches(existing_sources, right_ref) && refs_eq(left_ref, right_name) {
                return Some(GpuJoin {
                    left_table: right_ref.to_owned(),
                    left_col: right_qc.column,
                    right_table: left_ref.to_owned(),
                    right_col: left_qc.column,
                });
            }
        }

        // Unqualified columns: we don't know which column belongs to which
        // table at parse time. Store them without table qualifiers and let
        // the executor resolve them against actual pinned table schemas.
        if left_qc.table.is_none() && right_qc.table.is_none() {
            return Some(GpuJoin {
                left_table: String::new(), // unresolved — executor will match by column name
                left_col: left_qc.column,
                right_table: String::new(),
                right_col: right_qc.column,
            });
        }
    }
    // Handle AND (multiple join conditions — take the first equi-join)
    if let Expr::BinaryOp {
        left,
        op: BinaryOperator::And,
        right,
    } = expr
    {
        if let Some(join) = parse_equi_join_expr(left, existing_sources, right_source) {
            return Some(join);
        }
        return parse_equi_join_expr(right, existing_sources, right_source);
    }
    None
}

fn source_matches(sources: &[TableSource], reference: &str) -> bool {
    sources
        .iter()
        .any(|s| refs_eq(reference, s.alias.as_deref().unwrap_or(s.name.as_str())))
}

fn refs_eq(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

fn extract_qualified_column(expr: &Expr) -> Option<QualifiedColumn> {
    match expr {
        Expr::Identifier(ident) => Some(QualifiedColumn {
            table: None,
            column: ident.value.clone(),
        }),
        Expr::CompoundIdentifier(parts) => {
            if parts.len() == 2 {
                Some(QualifiedColumn {
                    table: Some(parts[0].value.clone()),
                    column: parts[1].value.clone(),
                })
            } else if parts.len() == 1 {
                Some(QualifiedColumn {
                    table: None,
                    column: parts[0].value.clone(),
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

fn build_alias_map(sources: &[TableSource]) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    for src in sources {
        let key = src
            .alias
            .as_deref()
            .unwrap_or(src.name.as_str())
            .to_lowercase();
        map.insert(key, src.name.to_lowercase());
    }
    map
}

fn parse_where_clause(
    expr: &Expr,
    alias_map: &std::collections::HashMap<String, String>,
    filters: &mut Vec<GpuFilter>,
    joins: &mut Vec<GpuJoin>,
) -> Option<()> {
    match expr {
        Expr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            parse_where_clause(left, alias_map, filters, joins)?;
            parse_where_clause(right, alias_map, filters, joins)?;
        }
        Expr::BinaryOp { left, op, right } => {
            // Check if both sides are column references (potential join condition).
            if let (Some(left_qc), Some(right_qc)) = (
                extract_qualified_column(left),
                extract_qualified_column(right),
            ) {
                if left_qc.table.is_some() && right_qc.table.is_some() && *op == BinaryOperator::Eq
                {
                    // Cross-table equality → implicit join condition.
                    let lt = left_qc.table.as_ref().unwrap();
                    let rt = right_qc.table.as_ref().unwrap();
                    let lt_resolved = alias_map
                        .get(&lt.to_lowercase())
                        .cloned()
                        .unwrap_or_else(|| lt.clone());
                    let rt_resolved = alias_map
                        .get(&rt.to_lowercase())
                        .cloned()
                        .unwrap_or_else(|| rt.clone());
                    if lt_resolved != rt_resolved {
                        joins.push(GpuJoin {
                            left_table: lt.clone(),
                            left_col: left_qc.column,
                            right_table: rt.clone(),
                            right_col: right_qc.column,
                        });
                        return Some(());
                    }
                }
            }
            // Single-table filter predicate.
            let compare_op = match op {
                BinaryOperator::Eq => ScalarCompareOp::Eq,
                BinaryOperator::Lt => ScalarCompareOp::Lt,
                BinaryOperator::LtEq => ScalarCompareOp::Le,
                BinaryOperator::Gt => ScalarCompareOp::Gt,
                BinaryOperator::GtEq => ScalarCompareOp::Ge,
                _ => return None,
            };
            let (qc, literal) = extract_qualified_column_and_literal(left, right)?;
            filters.push(GpuFilter {
                table: qc.table,
                column: qc.column,
                op: compare_op,
                literal,
            });
        }
        Expr::Between {
            expr: col_expr,
            negated,
            low,
            high,
        } if !negated => {
            let qc = extract_qualified_column(col_expr)?;
            let low_lit = extract_literal(low)?;
            let high_lit = extract_literal(high)?;
            filters.push(GpuFilter {
                table: qc.table.clone(),
                column: qc.column.clone(),
                op: ScalarCompareOp::Ge,
                literal: low_lit,
            });
            filters.push(GpuFilter {
                table: qc.table,
                column: qc.column,
                op: ScalarCompareOp::Le,
                literal: high_lit,
            });
        }
        Expr::Nested(inner) => {
            parse_where_clause(inner, alias_map, filters, joins)?;
        }
        _ => return None,
    }
    Some(())
}

fn extract_qualified_column_and_literal(
    left: &Expr,
    right: &Expr,
) -> Option<(QualifiedColumn, GpuLiteral)> {
    if let (Some(qc), Some(lit)) = (extract_qualified_column(left), extract_literal(right)) {
        return Some((qc, lit));
    }
    if let (Some(lit), Some(qc)) = (extract_literal(left), extract_qualified_column(right)) {
        return Some((qc, lit));
    }
    None
}

fn extract_column_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Identifier(ident) => Some(ident.value.clone()),
        Expr::CompoundIdentifier(parts) => Some(parts.last()?.value.clone()),
        _ => None,
    }
}

fn extract_literal(expr: &Expr) -> Option<GpuLiteral> {
    match expr {
        Expr::Value(val) => match &val.value {
            Value::Number(s, _) => {
                if let Ok(i) = s.parse::<i64>() {
                    Some(GpuLiteral::Int(i))
                } else if let Ok(f) = s.parse::<f64>() {
                    Some(GpuLiteral::Float(f))
                } else {
                    None
                }
            }
            Value::SingleQuotedString(s) => Some(GpuLiteral::Str(s.clone())),
            _ => None,
        },
        Expr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => {
            let lit = extract_literal(expr)?;
            Some(negate_literal(lit))
        }
        // DuckDB-style typed literals: DATE '1994-01-01' parsed as
        // TypedString by sqlparser. Convert to days-since-epoch in pure Rust.
        Expr::TypedString {
            data_type: ast::DataType::Date,
            value,
        } => {
            let s = match value {
                Value::SingleQuotedString(s) => s.as_str(),
                _ => return None,
            };
            let days = parse_date_to_days_since_epoch(s)?;
            Some(GpuLiteral::Date(days))
        }
        _ => None,
    }
}

/// Parse 'YYYY-MM-DD' to days since 1970-01-01 using civil date arithmetic.
fn parse_date_to_days_since_epoch(s: &str) -> Option<i32> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 3 {
        return None;
    }
    let year: i32 = parts[0].parse().ok()?;
    let month: u32 = parts[1].parse().ok()?;
    let day: u32 = parts[2].parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    // Civil date → days since epoch (same algorithm as tests/extension.rs date_days).
    let adjusted_year = year - i32::from(month <= 2);
    let era = if adjusted_year >= 0 {
        adjusted_year
    } else {
        adjusted_year - 399
    } / 400;
    let year_of_era = adjusted_year - era * 400;
    let month_offset = i32::try_from(month).ok()? + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * month_offset + 2) / 5 + i32::try_from(day).ok()? - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    Some(era * 146_097 + day_of_era - 719_468)
}

fn negate_literal(lit: GpuLiteral) -> GpuLiteral {
    match lit {
        GpuLiteral::Int(i) => GpuLiteral::Int(-i),
        GpuLiteral::Float(f) => GpuLiteral::Float(-f),
        GpuLiteral::Date(d) => GpuLiteral::Date(-d),
        other => other,
    }
}

fn parse_group_by(group_by: &GroupByExpr) -> Option<Vec<String>> {
    match group_by {
        GroupByExpr::All(_) => None,
        GroupByExpr::Expressions(exprs, _) => {
            let mut cols = Vec::new();
            for expr in exprs {
                cols.push(extract_column_name(expr)?);
            }
            Some(cols)
        }
    }
}

fn parse_select_items(
    items: &[SelectItem],
    group_by_columns: &[String],
) -> Option<(Vec<GpuProjection>, Vec<GpuAggExpr>)> {
    let mut projections = Vec::new();
    let mut agg_exprs = Vec::new();

    for item in items {
        match item {
            SelectItem::UnnamedExpr(expr) => {
                let (proj, maybe_agg) = parse_select_expr(expr, None, group_by_columns)?;
                projections.push(proj);
                if let Some(agg) = maybe_agg {
                    agg_exprs.push(agg);
                }
            }
            SelectItem::ExprWithAlias { expr, alias } => {
                let (proj, maybe_agg) =
                    parse_select_expr(expr, Some(alias.value.as_str()), group_by_columns)?;
                projections.push(proj);
                if let Some(agg) = maybe_agg {
                    agg_exprs.push(agg);
                }
            }
            SelectItem::Wildcard(_) => return None,
            _ => return None,
        }
    }

    Some((projections, agg_exprs))
}

#[allow(clippy::only_used_in_recursion)]
fn parse_select_expr(
    expr: &Expr,
    alias: Option<&str>,
    group_by_columns: &[String],
) -> Option<(GpuProjection, Option<GpuAggExpr>)> {
    // Direct column reference.
    if let Some(col) = extract_column_name(expr) {
        let name = alias.unwrap_or(col.as_str()).to_owned();
        return Some((
            GpuProjection {
                alias: name,
                expr: GpuExpr::Column(col),
            },
            None,
        ));
    }

    // Aggregate function call: SUM(expr), COUNT(*), etc.
    if let Expr::Function(func) = expr {
        let func_name = func.name.to_string().to_uppercase();
        let agg_func = match func_name.as_str() {
            "SUM" => AggFunc::Sum,
            "COUNT" => AggFunc::Count,
            "MIN" => AggFunc::Min,
            "MAX" => AggFunc::Max,
            "AVG" => AggFunc::Avg,
            _ => return None,
        };

        let agg_inner = parse_agg_arg(&func.args)?;
        let agg_alias = alias.map(|s| s.to_owned()).unwrap_or_else(|| {
            format!(
                "{}({})",
                func_name.to_lowercase(),
                format_gpu_expr(&agg_inner)
            )
        });

        return Some((
            GpuProjection {
                alias: agg_alias.clone(),
                expr: GpuExpr::AggCall {
                    func: agg_func,
                    arg: Box::new(agg_inner.clone()),
                },
            },
            Some(GpuAggExpr {
                func: agg_func,
                expr: agg_inner,
                alias: agg_alias,
            }),
        ));
    }

    // Binary arithmetic expression — recursive.
    if let Expr::BinaryOp { left, op, right } = expr {
        let gpu_op = match op {
            BinaryOperator::Multiply => GpuBinOp::Mul,
            BinaryOperator::Plus => GpuBinOp::Add,
            BinaryOperator::Minus => GpuBinOp::Sub,
            _ => return None,
        };
        let left_expr = parse_gpu_expr(left)?;
        let right_expr = parse_gpu_expr(right)?;
        let gpu_expr = GpuExpr::BinaryOp {
            op: gpu_op,
            left: Box::new(left_expr),
            right: Box::new(right_expr),
        };
        let expr_alias = alias
            .map(|s| s.to_owned())
            .unwrap_or_else(|| format_gpu_expr(&gpu_expr));
        return Some((
            GpuProjection {
                alias: expr_alias,
                expr: gpu_expr,
            },
            None,
        ));
    }

    // Nested expression.
    if let Expr::Nested(inner) = expr {
        return parse_select_expr(inner, alias, group_by_columns);
    }

    None
}

/// Recursively parse a SQL expression into a GpuExpr.
fn parse_gpu_expr(expr: &Expr) -> Option<GpuExpr> {
    // Column reference.
    if let Some(col) = extract_column_name(expr) {
        return Some(GpuExpr::Column(col));
    }
    // Literal.
    if let Some(lit) = extract_literal(expr) {
        return Some(GpuExpr::Literal(lit));
    }
    // Binary op — recursive.
    if let Expr::BinaryOp { left, op, right } = expr {
        let gpu_op = match op {
            BinaryOperator::Multiply => GpuBinOp::Mul,
            BinaryOperator::Plus => GpuBinOp::Add,
            BinaryOperator::Minus => GpuBinOp::Sub,
            _ => return None,
        };
        return Some(GpuExpr::BinaryOp {
            op: gpu_op,
            left: Box::new(parse_gpu_expr(left)?),
            right: Box::new(parse_gpu_expr(right)?),
        });
    }
    // Parenthesized.
    if let Expr::Nested(inner) = expr {
        return parse_gpu_expr(inner);
    }
    // Negative literal.
    if let Expr::UnaryOp {
        op: UnaryOperator::Minus,
        expr: inner,
    } = expr
    {
        if let Some(lit) = extract_literal(inner) {
            return Some(GpuExpr::Literal(negate_literal(lit)));
        }
    }
    None
}

fn format_gpu_expr(expr: &GpuExpr) -> String {
    match expr {
        GpuExpr::Column(c) => c.clone(),
        GpuExpr::Literal(GpuLiteral::Int(i)) => i.to_string(),
        GpuExpr::Literal(GpuLiteral::Float(f)) => f.to_string(),
        GpuExpr::Literal(GpuLiteral::Str(s)) => format!("'{s}'"),
        GpuExpr::Literal(GpuLiteral::Date(d)) => format!("date({d})"),
        GpuExpr::BinaryOp { op, left, right } => {
            let op_str = match op {
                GpuBinOp::Add => "+",
                GpuBinOp::Sub => "-",
                GpuBinOp::Mul => "*",
            };
            format!(
                "({} {} {})",
                format_gpu_expr(left),
                op_str,
                format_gpu_expr(right)
            )
        }
        GpuExpr::AggCall { func, arg } => {
            let func_str = match func {
                AggFunc::Sum => "sum",
                AggFunc::Count => "count",
                AggFunc::Min => "min",
                AggFunc::Max => "max",
                AggFunc::Avg => "avg",
            };
            format!("{}({})", func_str, format_gpu_expr(arg))
        }
    }
}

/// Parse the argument to an aggregate function. Returns a `GpuExpr` representing
/// the argument (column, expression, or a dummy Column("*") for COUNT(*)).
fn parse_agg_arg(args: &ast::FunctionArguments) -> Option<GpuExpr> {
    match args {
        ast::FunctionArguments::None => Some(GpuExpr::Column("*".to_owned())),
        ast::FunctionArguments::List(arg_list) => {
            if arg_list.args.is_empty() {
                return Some(GpuExpr::Column("*".to_owned()));
            }
            if arg_list.args.len() != 1 {
                return None;
            }
            match &arg_list.args[0] {
                FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => {
                    Some(GpuExpr::Column("*".to_owned()))
                }
                FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => parse_gpu_expr(expr),
                _ => None,
            }
        }
        _ => None,
    }
}

fn parse_order_by(order_by: &ast::OrderBy) -> Option<Vec<GpuOrderBy>> {
    match &order_by.kind {
        OrderByKind::All(_) => None,
        OrderByKind::Expressions(exprs) => {
            let mut result = Vec::new();
            for order_expr in exprs {
                let column = extract_column_name(&order_expr.expr)?;
                let descending = order_expr.options.asc.map(|asc| !asc).unwrap_or(false);
                result.push(GpuOrderBy { column, descending });
            }
            Some(result)
        }
    }
}

fn parse_limit(limit: &Option<Expr>) -> Option<Option<usize>> {
    match limit {
        None => Some(None),
        Some(expr) => match expr {
            Expr::Value(val) => match &val.value {
                Value::Number(s, _) => Some(Some(s.parse::<usize>().ok()?)),
                _ => None,
            },
            _ => None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_select() {
        let plan = try_build_gpu_plan("SELECT value FROM numbers WHERE value > 5").unwrap();
        assert_eq!(plan.sources[0].name, "numbers");
        assert_eq!(plan.filters.len(), 1);
        assert_eq!(plan.filters[0].column, "value");
        assert!(matches!(plan.filters[0].op, ScalarCompareOp::Gt));
        assert_eq!(plan.projections.len(), 1);
        assert!(plan.aggregate.is_none());
    }

    #[test]
    fn parses_multi_predicate_where() {
        let plan = try_build_gpu_plan("SELECT x FROM t WHERE x > 10 AND y < 20 AND z = 5").unwrap();
        assert_eq!(plan.filters.len(), 3);
    }

    #[test]
    fn parses_between() {
        let plan = try_build_gpu_plan("SELECT x FROM t WHERE x BETWEEN 5 AND 10").unwrap();
        assert_eq!(plan.filters.len(), 2);
        assert!(matches!(plan.filters[0].op, ScalarCompareOp::Ge));
        assert!(matches!(plan.filters[1].op, ScalarCompareOp::Le));
    }

    #[test]
    fn parses_aggregate_with_expression_arg() {
        let plan = try_build_gpu_plan(
            "SELECT SUM(price * discount) AS revenue FROM lineitem WHERE qty < 24",
        )
        .unwrap();
        assert!(plan.aggregate.is_some());
        let agg = plan.aggregate.as_ref().unwrap();
        assert!(agg.group_by.is_empty());
        assert_eq!(agg.agg_exprs.len(), 1);
        assert!(matches!(agg.agg_exprs[0].func, AggFunc::Sum));
        // The inner expression should be a BinaryOp(Mul, Column, Column).
        assert!(matches!(
            &agg.agg_exprs[0].expr,
            GpuExpr::BinaryOp {
                op: GpuBinOp::Mul,
                ..
            }
        ));
    }

    #[test]
    fn parses_simple_aggregate() {
        let plan =
            try_build_gpu_plan("SELECT SUM(price) AS total FROM lineitem WHERE qty < 24").unwrap();
        assert!(plan.aggregate.is_some());
        let agg = plan.aggregate.as_ref().unwrap();
        assert!(agg.group_by.is_empty());
        assert_eq!(agg.agg_exprs.len(), 1);
        assert!(matches!(agg.agg_exprs[0].func, AggFunc::Sum));
    }

    #[test]
    fn parses_grouped_aggregate() {
        let plan = try_build_gpu_plan(
            "SELECT key, SUM(value) AS total, COUNT(*) AS cnt \
             FROM data GROUP BY key ORDER BY key",
        )
        .unwrap();
        let agg = plan.aggregate.as_ref().unwrap();
        assert_eq!(agg.group_by, vec!["key"]);
        assert_eq!(agg.agg_exprs.len(), 2);
        assert_eq!(plan.order_by.len(), 1);
        assert!(!plan.order_by[0].descending);
    }

    #[test]
    fn parses_order_by_desc_with_limit() {
        let plan = try_build_gpu_plan("SELECT x, y FROM t ORDER BY x DESC, y LIMIT 10").unwrap();
        assert_eq!(plan.order_by.len(), 2);
        assert!(plan.order_by[0].descending);
        assert!(!plan.order_by[1].descending);
        assert_eq!(plan.limit, Some(10));
    }

    #[test]
    fn parses_join() {
        let plan = try_build_gpu_plan("SELECT a FROM t1 JOIN t2 ON t1.id = t2.id").unwrap();
        assert_eq!(plan.sources.len(), 2);
        assert_eq!(plan.sources[0].name, "t1");
        assert_eq!(plan.sources[1].name, "t2");
        assert_eq!(plan.joins.len(), 1);
        assert_eq!(plan.joins[0].left_col, "id");
        assert_eq!(plan.joins[0].right_col, "id");
    }

    #[test]
    fn parses_three_table_join() {
        let plan = try_build_gpu_plan(
            "SELECT a FROM t1 JOIN t2 ON t1.id = t2.id JOIN t3 ON t2.key = t3.key",
        )
        .unwrap();
        assert_eq!(plan.sources.len(), 3);
        assert_eq!(plan.joins.len(), 2);
    }

    #[test]
    fn parses_comma_join_from_where() {
        let plan =
            try_build_gpu_plan("SELECT a FROM t1, t2 WHERE t1.id = t2.id AND t1.x > 5").unwrap();
        assert_eq!(plan.sources.len(), 2);
        assert_eq!(plan.joins.len(), 1);
        assert_eq!(plan.filters.len(), 1);
    }

    #[test]
    fn rejects_subquery() {
        assert!(try_build_gpu_plan("SELECT a FROM (SELECT * FROM t)").is_none());
    }

    #[test]
    fn rejects_cte() {
        assert!(try_build_gpu_plan("WITH cte AS (SELECT 1) SELECT * FROM cte").is_none());
    }

    #[test]
    fn rejects_or_predicate() {
        assert!(try_build_gpu_plan("SELECT a FROM t WHERE a > 5 OR a < 1").is_none());
    }

    #[test]
    fn parses_string_literal() {
        let plan = try_build_gpu_plan("SELECT a FROM t WHERE name = 'foo'").unwrap();
        assert_eq!(plan.filters.len(), 1);
        assert!(matches!(plan.filters[0].literal, GpuLiteral::Str(ref s) if s == "foo"));
    }

    #[test]
    fn parses_date_literal() {
        let plan = try_build_gpu_plan("SELECT a FROM t WHERE d > DATE '2024-01-01'").unwrap();
        assert_eq!(plan.filters.len(), 1);
        // 2024-01-01 = 19723 days since epoch
        assert!(matches!(plan.filters[0].literal, GpuLiteral::Date(19723)));
    }

    #[test]
    fn parses_date_epoch() {
        let plan = try_build_gpu_plan("SELECT a FROM t WHERE d = DATE '1970-01-01'").unwrap();
        assert!(matches!(plan.filters[0].literal, GpuLiteral::Date(0)));
    }

    #[test]
    fn parses_q3_full_sql() {
        let sql = "SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue, \
                   o_orderdate, o_shippriority \
                   FROM customer \
                   JOIN orders ON c_custkey = o_custkey \
                   JOIN lineitem ON l_orderkey = o_orderkey \
                   WHERE c_mktsegment = 'BUILDING' \
                   AND o_orderdate < DATE '1995-03-25' \
                   AND l_shipdate > DATE '1995-03-25' \
                   GROUP BY l_orderkey, o_orderdate, o_shippriority \
                   ORDER BY revenue DESC, o_orderdate \
                   LIMIT 10";
        let plan = try_build_gpu_plan(sql);
        assert!(plan.is_some(), "Q3 SQL should parse into a GpuPlan");
        let plan = plan.unwrap();
        assert_eq!(plan.sources.len(), 3);
        assert_eq!(plan.joins.len(), 2);
        assert_eq!(plan.filters.len(), 3);
        assert!(plan.aggregate.is_some());
        let agg = plan.aggregate.as_ref().unwrap();
        assert_eq!(agg.group_by.len(), 3);
        assert_eq!(agg.agg_exprs.len(), 1);
        assert_eq!(plan.order_by.len(), 2);
        assert_eq!(plan.limit, Some(10));
    }

    #[test]
    fn parses_q3_benchmark_sql() {
        // The actual benchmark Q3 SQL uses comma-FROM and table aliases.
        let sql = "select l.l_orderkey, sum(l.l_extendedprice * (1 - l.l_discount)) as revenue, \
                   o.o_orderdate, o.o_shippriority \
                   from customer c, orders o, lineitem l \
                   where c.c_mktsegment = 'HOUSEHOLD' \
                   and c.c_custkey = o.o_custkey \
                   and l.l_orderkey = o.o_orderkey \
                   and o.o_orderdate < date '1995-03-25' \
                   and l.l_shipdate > date '1995-03-25' \
                   group by l.l_orderkey, o.o_orderdate, o.o_shippriority \
                   order by revenue desc, o.o_orderdate \
                   limit 10";
        let plan = try_build_gpu_plan(sql);
        assert!(
            plan.is_some(),
            "Q3 benchmark SQL should parse into a GpuPlan"
        );
        let plan = plan.unwrap();
        assert_eq!(plan.sources.len(), 3);
        assert_eq!(plan.joins.len(), 2); // extracted from WHERE clause
        assert_eq!(plan.filters.len(), 3); // c_mktsegment, o_orderdate, l_shipdate
        assert!(plan.aggregate.is_some());
    }

    #[test]
    fn parses_q3_compound_expression() {
        let plan = try_build_gpu_plan(
            "SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue \
             FROM lineitem GROUP BY l_orderkey",
        )
        .unwrap();
        assert!(plan.aggregate.is_some());
        let agg = plan.aggregate.as_ref().unwrap();
        assert_eq!(agg.group_by, vec!["l_orderkey"]);
        assert_eq!(agg.agg_exprs.len(), 1);
        // SUM(l_extendedprice * (1 - l_discount)) →
        //   AggCall(Sum, BinaryOp(Mul, Column, BinaryOp(Sub, Literal(1), Column)))
        match &agg.agg_exprs[0].expr {
            GpuExpr::BinaryOp {
                op: GpuBinOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(left.as_ref(), GpuExpr::Column(c) if c == "l_extendedprice"));
                assert!(matches!(
                    right.as_ref(),
                    GpuExpr::BinaryOp {
                        op: GpuBinOp::Sub,
                        ..
                    }
                ));
            }
            other => panic!("expected BinaryOp(Mul, ...), got: {other:?}"),
        }
    }

    fn test_pinned_table(columns: &[(&str, PinnedColumn)]) -> Arc<PinnedTable> {
        Arc::new(PinnedTable {
            columns: columns
                .iter()
                .map(|(name, column)| ((*name).to_owned(), column.clone()))
                .collect(),
            column_meta: HashMap::new(),
            row_count: 1,
        })
    }

    #[test]
    fn join_liveness_drops_dead_dimension_sources() {
        let sql = "select l.l_orderkey, sum(l.l_extendedprice * (1 - l.l_discount)) as revenue, \
                   o.o_orderdate, o.o_shippriority \
                   from customer c, orders o, lineitem l \
                   where c.c_mktsegment = 'HOUSEHOLD' \
                   and c.c_custkey = o.o_custkey \
                   and l.l_orderkey = o.o_orderkey \
                   and o.o_orderdate < date '1995-03-25' \
                   and l.l_shipdate > date '1995-03-25' \
                   group by l.l_orderkey, o.o_orderdate, o.o_shippriority \
                   order by revenue desc, o.o_orderdate \
                   limit 10";
        let plan = try_build_gpu_plan(sql).unwrap();
        let pinned_tables = vec![
            test_pinned_table(&[
                ("c_custkey", PinnedColumn::Int32(vec![1])),
                (
                    "c_mktsegment",
                    PinnedColumn::DictionaryI32 {
                        codes: vec![0],
                        values: vec!["HOUSEHOLD".to_owned()],
                        reverse: HashMap::from([("HOUSEHOLD".to_owned(), 0)]),
                    },
                ),
            ]),
            test_pinned_table(&[
                ("o_custkey", PinnedColumn::Int32(vec![1])),
                ("o_orderkey", PinnedColumn::Int32(vec![1])),
                ("o_orderdate", PinnedColumn::Date32(vec![9199])),
                ("o_shippriority", PinnedColumn::Int32(vec![0])),
            ]),
            test_pinned_table(&[
                ("l_orderkey", PinnedColumn::Int32(vec![1])),
                ("l_extendedprice", PinnedColumn::Float32(vec![1.0])),
                ("l_discount", PinnedColumn::Float32(vec![0.0])),
                ("l_shipdate", PinnedColumn::Date32(vec![9201])),
            ]),
        ];
        let execution_order = vec![0, 1, 2];
        let steps = join_execution_steps(&plan, &pinned_tables, &execution_order).unwrap();
        let live_after =
            live_sources_after_join_steps(&plan, &pinned_tables, &execution_order, &steps, &[]);

        assert_eq!(live_after.len(), 2);
        assert!(
            !live_after[0][0],
            "customer should be dead after customer->orders"
        );
        assert!(live_after[0][1], "orders stays live for final output");
        assert!(live_after[1][1], "orders stays live through the final join");
        assert!(live_after[1][2], "lineitem stays live for grouping and sum");
    }

    #[test]
    fn join_liveness_keeps_a_cardinality_carrier() {
        let plan = try_build_gpu_plan(
            "select count(*) \
             from t1 join t2 on t1.id = t2.id join t3 on t2.key = t3.key",
        )
        .unwrap();
        let pinned_tables = vec![
            test_pinned_table(&[("id", PinnedColumn::Int32(vec![1]))]),
            test_pinned_table(&[
                ("id", PinnedColumn::Int32(vec![1])),
                ("key", PinnedColumn::Int32(vec![1])),
            ]),
            test_pinned_table(&[("key", PinnedColumn::Int32(vec![1]))]),
        ];
        let execution_order = vec![0, 1, 2];
        let steps = join_execution_steps(&plan, &pinned_tables, &execution_order).unwrap();
        let live_after =
            live_sources_after_join_steps(&plan, &pinned_tables, &execution_order, &steps, &[]);

        assert!(
            live_after[0][1],
            "the first probe stays live to carry cardinality"
        );
        assert!(
            live_after[1][2],
            "the final probe stays live to carry cardinality"
        );
    }

    #[test]
    fn dense_lookup_join_strategy_is_shape_based() {
        let dense_table = PinnedTable {
            columns: HashMap::new(),
            column_meta: HashMap::from([(
                "id".to_owned(),
                crate::pinned::ColumnMeta {
                    stats: crate::pinned::ColumnStats::Integer {
                        min: 1,
                        max: 6_000,
                        distinct_count: 5_000,
                    },
                    pilot_kind: crate::pinned::PilotColumnKind::Int32,
                },
            )]),
            row_count: 5_000,
        };
        let wide_table = PinnedTable {
            columns: HashMap::new(),
            column_meta: HashMap::from([(
                "id".to_owned(),
                crate::pinned::ColumnMeta {
                    stats: crate::pinned::ColumnStats::Integer {
                        min: 1,
                        max: 500_000,
                        distinct_count: 5_000,
                    },
                    pilot_kind: crate::pinned::PilotColumnKind::Int32,
                },
            )]),
            row_count: 5_000,
        };

        assert!(matches!(
            choose_gpu_join_strategy(&dense_table, "id", 5_000).unwrap(),
            GpuJoinStrategy::DenseLookup { .. }
        ));
        assert!(matches!(
            choose_gpu_join_strategy(&wide_table, "id", 5_000).unwrap(),
            GpuJoinStrategy::Hash { .. }
        ));
    }
}

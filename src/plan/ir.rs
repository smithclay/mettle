//! GPU plan IR data types. Pure data — no logic other than derive-likes.

use crate::metal::ScalarCompareOp;

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QualifiedColumn {
    pub table: Option<String>,
    pub column: String,
}

impl QualifiedColumn {
    pub fn is_wildcard(&self) -> bool {
        self.table.is_none() && self.column == "*"
    }

    pub fn display_name(&self) -> String {
        match &self.table {
            Some(table) => format!("{table}.{}", self.column),
            None => self.column.clone(),
        }
    }
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
    Column(QualifiedColumn),
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
    pub group_by: Vec<QualifiedColumn>,
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
pub(super) enum AggregateStateKind {
    CountU64,
    SumF64,
    MinF64,
    MaxF64,
    AvgSumCount,
}

#[derive(Debug, Clone)]
pub(super) enum AggregateInputPlan {
    CountStar,
    Expr(GpuExpr),
}

#[derive(Debug, Clone)]
pub(super) struct AggregateSlotPlan {
    pub(super) alias: String,
    pub(super) func: AggFunc,
    pub(super) input: AggregateInputPlan,
    pub(super) state_kind: AggregateStateKind,
}

#[derive(Debug, Clone)]
pub(super) enum GroupedAggregateOutput {
    GroupKey { group_idx: usize, alias: String },
    Aggregate { agg_idx: usize },
}

#[derive(Debug, Clone)]
pub(super) struct GroupedAggregatePlan {
    pub(super) group_by: Vec<QualifiedColumn>,
    pub(super) aggregate_slots: Vec<AggregateSlotPlan>,
    pub(super) outputs: Vec<GroupedAggregateOutput>,
}

#[derive(Debug, Clone)]
pub(super) struct SourceSelection {
    pub(super) local_indices: Vec<usize>,
    pub(super) final_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(super) struct ResolvedJoinEdge {
    pub(super) left_idx: usize,
    pub(super) left_col: String,
    pub(super) right_idx: usize,
    pub(super) right_col: String,
}

#[derive(Debug, Clone)]
pub(super) struct JoinExecutionStep {
    pub(super) build_source_idx: usize,
    pub(super) probe_source_idx: usize,
    pub(super) build_col_name: String,
    pub(super) probe_col_name: String,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum GpuJoinStrategy {
    Hash { ht_capacity: u32 },
    DenseLookup { min_key: i32, slot_count: u32 },
}

#[derive(Debug, Clone)]
pub struct GpuOrderBy {
    pub column: String,
    pub descending: bool,
}

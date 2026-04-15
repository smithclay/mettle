//! Column/expression resolution over a `GpuPlan` + `PinnedTable`s, plus the
//! predicate-transfer pass that prunes source indices across join edges.

use std::collections::HashSet;
use std::sync::Arc;

use crate::pinned::{PinnedColumn, PinnedTable};

use super::super::helpers::{column_name_only, refs_eq};
use super::super::ir::{GpuBinOp, GpuExpr, GpuFilter, GpuLiteral, GpuPlan, ResolvedJoinEdge};

pub(super) fn resolve_column_in_sources<'a>(
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

pub(super) fn resolve_column_name_in_sources<'a>(
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

pub(super) fn resolve_join_endpoint(
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

pub(super) fn find_equivalent_column_on_table(
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

pub(super) fn source_filters<'a>(
    plan: &'a GpuPlan,
    table: &PinnedTable,
    src_ref: &str,
) -> Vec<&'a GpuFilter> {
    plan.filters
        .iter()
        .filter(|filter| match filter.table.as_deref() {
            Some(table_ref) => refs_eq(table_ref, src_ref),
            None => table.column(&filter.column).is_some(),
        })
        .collect()
}

pub(super) fn apply_predicate_transfer(
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

pub(super) fn resolved_join_edges(
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

pub(super) fn collect_join_key_set(col: &PinnedColumn, indices: &[usize]) -> Option<HashSet<i32>> {
    indices
        .iter()
        .map(|&row_idx| join_key_value(col, row_idx))
        .collect::<Option<HashSet<_>>>()
}

pub(super) fn join_key_value(col: &PinnedColumn, row_idx: usize) -> Option<i32> {
    match col {
        PinnedColumn::Int32(values) | PinnedColumn::Date32(values) => Some(values[row_idx]),
        PinnedColumn::Int64(values) => i32::try_from(values[row_idx]).ok(),
        PinnedColumn::Float32(_)
        | PinnedColumn::Float64(_)
        | PinnedColumn::DictionaryI32 { .. } => None,
    }
}

pub(super) fn fused_mul_one_minus_columns(expr: &GpuExpr) -> Option<(&str, &str)> {
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

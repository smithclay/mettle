//! SQL AST → `GpuPlan` IR. No GPU or execution logic lives here.

use sqlparser::ast::{
    self, BinaryOperator, Expr, FunctionArg, FunctionArgExpr, GroupByExpr, ObjectName, OrderByKind,
    SelectItem, SetExpr, Statement, UnaryOperator, Value,
};
use sqlparser::dialect::DuckDbDialect;
use sqlparser::parser::Parser;

use crate::metal::ScalarCompareOp;

use super::ir::{
    AggFunc, GpuAggExpr, GpuAggregate, GpuBinOp, GpuExpr, GpuFilter, GpuJoin, GpuLiteral,
    GpuOrderBy, GpuPlan, GpuProjection, QualifiedColumn, TableSource,
};

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

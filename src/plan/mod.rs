//! SQL → GPU plan builder and executor.
//!
//! Parses user SQL via `sqlparser` into a flat `GpuPlan` pipeline IR
//! (scan → filter → join → project → aggregate → sort → limit), then
//! executes it against GPU-resident pinned tables on Metal with late
//! materialization (index arrays through joins, data gathered at output).
//! Returns `None` from `try_build_gpu_plan` for unsupported query shapes,
//! allowing fallthrough to DuckDB CPU execution.

mod aggregate;
mod executor;
mod helpers;
mod ir;
mod parser;

pub use executor::{result_cache_key, try_execute};
#[allow(unused_imports)]
pub use ir::{
    AggFunc, GpuAggExpr, GpuAggregate, GpuBinOp, GpuExpr, GpuFilter, GpuJoin, GpuLiteral,
    GpuOrderBy, GpuPlan, GpuProjection, QualifiedColumn, TableSource,
};
pub use parser::try_build_gpu_plan;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::metal::ScalarCompareOp;
    use crate::pinned::{PinnedColumn, PinnedTable};

    use super::executor::{
        choose_gpu_join_strategy, join_execution_steps, live_sources_after_join_steps,
    };
    use super::ir::GpuJoinStrategy;
    use super::{AggFunc, GpuBinOp, GpuExpr, GpuLiteral, try_build_gpu_plan};

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

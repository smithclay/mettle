use std::sync::{Mutex, MutexGuard, OnceLock};

use duckdb::Connection;

use mettle::{last_execution_mode, register_extension, reset_runtime_state_for_tests};

#[cfg(target_os = "macos")]
#[test]
fn gpu_processing_requires_initialization() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();

    let result = connection
        .prepare("CALL gpu_processing('SELECT 42')")
        .and_then(|mut stmt| {
            let mut rows = stmt.query([])?;
            rows.next()?;
            Ok(())
        });
    let error = result.expect_err("expected error from gpu_processing before init");
    assert!(
        error
            .to_string()
            .contains("call gpu_buffer_init before gpu_processing"),
        "unexpected error: {error}"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn gpu_register_table_pins_and_queries_via_generic_plan() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE numbers(value INTEGER);
             INSERT INTO numbers VALUES (1), (2), (3);",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();

    // Register returns table info.
    let mut stmt = connection
        .prepare("CALL gpu_register_table('numbers')")
        .unwrap();
    let mut rows = stmt.query([]).unwrap();
    let row = rows.next().unwrap().unwrap();
    let name: String = row.get(0).unwrap();
    let row_count: i64 = row.get(1).unwrap();
    assert_eq!(name, "numbers");
    assert_eq!(row_count, 3);
}

#[cfg(target_os = "macos")]
#[test]
fn generic_plan_executes_filter_on_pinned_table() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE numbers(value INTEGER);
             INSERT INTO numbers VALUES (1), (2), (3), (4), (5);",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('numbers')")
        .unwrap();

    let mut stmt = connection
        .prepare("CALL gpu_processing('SELECT value FROM numbers WHERE value > 2')")
        .unwrap();
    let mut rows = stmt.query([]).unwrap();

    let mut values: Vec<i32> = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        values.push(row.get(0).unwrap());
    }
    values.sort();
    assert_eq!(values, vec![3, 4, 5]);
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));
}

#[cfg(target_os = "macos")]
#[test]
fn generic_plan_executes_projection_with_filter() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE numbers(value INTEGER);
             INSERT INTO numbers VALUES (1), (2), (3);",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('numbers')")
        .unwrap();

    let mut stmt = connection
        .prepare(
            "CALL gpu_processing('SELECT value, value * 10 AS scaled FROM numbers WHERE value >= 2')",
        )
        .unwrap();
    let mut rows = stmt.query([]).unwrap();

    let mut result: Vec<(i32, i32)> = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        result.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }
    result.sort();
    assert_eq!(result, vec![(2, 20), (3, 30)]);
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));
}

#[cfg(target_os = "macos")]
#[test]
fn generic_plan_executes_multi_predicate_filter() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE data(x INTEGER, y INTEGER);
             INSERT INTO data VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50);",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('data')")
        .unwrap();

    let mut stmt = connection
        .prepare("CALL gpu_processing('SELECT x, y FROM data WHERE x > 1 AND y < 50')")
        .unwrap();
    let mut rows = stmt.query([]).unwrap();

    let mut result: Vec<(i32, i32)> = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        result.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }
    result.sort();
    assert_eq!(result, vec![(2, 20), (3, 30), (4, 40)]);
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));
}

#[cfg(target_os = "macos")]
#[test]
fn generic_plan_executes_order_by_desc_with_limit() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE numbers(value INTEGER);
             INSERT INTO numbers VALUES (5), (3), (1), (4), (2);",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('numbers')")
        .unwrap();

    let mut stmt = connection
        .prepare("CALL gpu_processing('SELECT value FROM numbers ORDER BY value DESC LIMIT 3')")
        .unwrap();
    let mut rows = stmt.query([]).unwrap();

    let mut values: Vec<i32> = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        values.push(row.get(0).unwrap());
    }
    assert_eq!(values, vec![5, 4, 3]);
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));
}

#[cfg(target_os = "macos")]
#[test]
fn generic_plan_filters_by_string_and_date() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE events(id INTEGER, category VARCHAR, event_date DATE);
             INSERT INTO events VALUES
               (1, 'A', DATE '2024-01-15'),
               (2, 'B', DATE '2024-02-20'),
               (3, 'A', DATE '2024-03-10'),
               (4, 'A', DATE '2024-04-05'),
               (5, 'B', DATE '2024-05-01');",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('events')")
        .unwrap();

    let mut stmt = connection
        .prepare(
            "CALL gpu_processing('SELECT id FROM events WHERE category = ''A'' AND event_date > DATE ''2024-02-01''')",
        )
        .unwrap();
    let mut rows = stmt.query([]).unwrap();

    let mut ids: Vec<i32> = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        ids.push(row.get(0).unwrap());
    }
    ids.sort();
    assert_eq!(ids, vec![3, 4]);
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));
}

#[cfg(target_os = "macos")]
#[test]
fn generic_plan_executes_ungrouped_aggregate() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE prices(value INTEGER);
             INSERT INTO prices VALUES (10), (20), (30), (40), (50);",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('prices')")
        .unwrap();

    let mut stmt = connection
        .prepare(
            "CALL gpu_processing('SELECT SUM(value) AS total, COUNT(*) AS cnt, MIN(value) AS lo, MAX(value) AS hi FROM prices WHERE value >= 20')",
        )
        .unwrap();
    let mut rows = stmt.query([]).unwrap();
    let row = rows.next().unwrap().unwrap();

    let total: f64 = row.get(0).unwrap();
    let cnt: f64 = row.get(1).unwrap();
    let lo: f64 = row.get(2).unwrap();
    let hi: f64 = row.get(3).unwrap();

    assert_eq!(total, 140.0);
    assert_eq!(cnt, 4.0);
    assert_eq!(lo, 20.0);
    assert_eq!(hi, 50.0);
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));
}

#[cfg(target_os = "macos")]
#[test]
fn generic_plan_executes_two_table_join() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE departments(dept_id INTEGER, dept_name VARCHAR);
             INSERT INTO departments VALUES (1, 'Engineering'), (2, 'Sales'), (3, 'HR');
             CREATE TABLE employees(emp_id INTEGER, dept_id INTEGER, salary INTEGER);
             INSERT INTO employees VALUES (10, 1, 100), (11, 1, 200), (12, 2, 150), (13, 3, 300);",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('departments')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('employees')")
        .unwrap();

    let mut stmt = connection
        .prepare(
            "CALL gpu_processing('SELECT e.emp_id, e.salary FROM departments d JOIN employees e ON d.dept_id = e.dept_id WHERE d.dept_name = ''Engineering''')",
        )
        .unwrap();
    let mut rows = stmt.query([]).unwrap();

    let mut result: Vec<(i32, i32)> = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        result.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }
    result.sort();
    assert_eq!(result, vec![(10, 100), (11, 200)]);
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));
}

#[cfg(target_os = "macos")]
#[test]
fn generic_plan_evaluates_compound_expression() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE items(price DOUBLE, discount DOUBLE);
             INSERT INTO items VALUES (100.0, 0.05), (200.0, 0.10), (50.0, 0.0);",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('items')")
        .unwrap();

    // Test: price * (1 - discount) as a compound expression
    let mut stmt = connection
        .prepare("CALL gpu_processing('SELECT price * (1 - discount) AS net FROM items')")
        .unwrap();
    let mut rows = stmt.query([]).unwrap();

    let mut values: Vec<f64> = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        values.push(row.get(0).unwrap());
    }
    // 100*(1-0.05)=95, 200*(1-0.10)=180, 50*(1-0.0)=50
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(values.len(), 3);
    assert!((values[0] - 50.0).abs() < 0.1);
    assert!((values[1] - 95.0).abs() < 0.1);
    assert!((values[2] - 180.0).abs() < 0.1);
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));
}

#[cfg(target_os = "macos")]
#[test]
fn generic_plan_evaluates_ungrouped_agg_with_expression() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE items(price DOUBLE, discount DOUBLE);
             INSERT INTO items VALUES (100.0, 0.05), (200.0, 0.10), (50.0, 0.0);",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('items')")
        .unwrap();

    // SUM(price * (1 - discount)) = 95 + 180 + 50 = 325
    let mut stmt = connection
        .prepare("CALL gpu_processing('SELECT SUM(price * (1 - discount)) AS revenue FROM items')")
        .unwrap();
    let mut rows = stmt.query([]).unwrap();
    let row = rows.next().unwrap().unwrap();
    let revenue: f64 = row.get(0).unwrap();
    assert!((revenue - 325.0).abs() < 1.0);
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));
}

#[cfg(target_os = "macos")]
#[test]
fn generic_plan_executes_q3_three_table_join_with_grouped_agg() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE customer(c_custkey INTEGER, c_mktsegment VARCHAR);
             INSERT INTO customer VALUES (1, 'BUILDING'), (2, 'HOUSEHOLD'), (3, 'BUILDING');
             CREATE TABLE orders(o_orderkey INTEGER, o_custkey INTEGER, o_orderdate DATE, o_shippriority INTEGER);
             INSERT INTO orders VALUES
               (100, 1, DATE '1995-03-10', 0),
               (101, 2, DATE '1995-03-20', 1),
               (102, 3, DATE '1995-03-05', 0),
               (103, 1, DATE '1995-04-01', 2);
             CREATE TABLE lineitem(l_orderkey INTEGER, l_extendedprice DOUBLE, l_discount DOUBLE, l_shipdate DATE);
             INSERT INTO lineitem VALUES
               (100, 1000.0, 0.05, DATE '1995-03-30'),
               (100, 500.0,  0.10, DATE '1995-04-01'),
               (101, 200.0,  0.00, DATE '1995-03-28'),
               (102, 300.0,  0.20, DATE '1995-03-26'),
               (103, 800.0,  0.15, DATE '1995-03-30');",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('customer')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('orders')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('lineitem')")
        .unwrap();

    // Q3-shaped query: 3-table join, string filter, date filters, grouped aggregate with expression
    let mut stmt = connection
        .prepare(
            "CALL gpu_processing('SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue, \
             o_orderdate, o_shippriority \
             FROM customer \
             JOIN orders ON c_custkey = o_custkey \
             JOIN lineitem ON l_orderkey = o_orderkey \
             WHERE c_mktsegment = ''BUILDING'' \
             AND o_orderdate < DATE ''1995-03-25'' \
             AND l_shipdate > DATE ''1995-03-25'' \
             GROUP BY l_orderkey, o_orderdate, o_shippriority \
             ORDER BY revenue DESC \
             LIMIT 10')",
        )
        .unwrap();
    let mut rows = stmt.query([]).unwrap();

    // Expected:
    // customer BUILDING: custkey 1 and 3
    // orders with o_orderdate < 1995-03-25: order 100 (custkey 1), 102 (custkey 3)
    // lineitem with l_shipdate > 1995-03-25:
    //   order 100: price=1000,disc=0.05 → 950.0 AND price=500,disc=0.10 → 450.0
    //   order 102: price=300,disc=0.20 → 240.0
    // Grouped:
    //   l_orderkey=100: revenue = 950+450 = 1400, o_orderdate=1995-03-10, shippriority=0
    //   l_orderkey=102: revenue = 240, o_orderdate=1995-03-05, shippriority=0
    // ORDER BY revenue DESC → 100 first, then 102

    let mut result: Vec<(f64, f64)> = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        let orderkey: f64 = row.get(0).unwrap();
        let revenue: f64 = row.get(1).unwrap();
        result.push((orderkey, revenue));
    }

    assert_eq!(result.len(), 2);
    // First row: orderkey 100, revenue ~1400
    assert!((result[0].0 - 100.0).abs() < 0.1);
    assert!((result[0].1 - 1400.0).abs() < 1.0);
    // Second row: orderkey 102, revenue ~240
    assert!((result[1].0 - 102.0).abs() < 0.1);
    assert!((result[1].1 - 240.0).abs() < 1.0);
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));
}

#[cfg(target_os = "macos")]
#[test]
fn generic_plan_executes_simple_three_table_grouped_aggregate() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE regions(region_id INTEGER, market INTEGER);
             INSERT INTO regions VALUES (10, 1), (20, 2);
             CREATE TABLE departments(dept_id INTEGER, region_id INTEGER);
             INSERT INTO departments VALUES (1, 10), (2, 10), (3, 20);
             CREATE TABLE employees(emp_id INTEGER, dept_id INTEGER, salary DOUBLE);
             INSERT INTO employees VALUES
                 (100, 1, 100.0), (101, 1, 150.0), (102, 2, 200.0),
                 (103, 3, 90.0), (104, 3, 110.0);",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('regions')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('departments')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('employees')")
        .unwrap();

    let mut stmt = connection
        .prepare(
            "CALL gpu_processing('SELECT r.market, SUM(e.salary) AS payroll
                                  FROM regions r
                                  JOIN departments d ON r.region_id = d.region_id
                                  JOIN employees e ON d.dept_id = e.dept_id
                                  GROUP BY r.market
                                  ORDER BY r.market')",
        )
        .unwrap();
    let mut rows = stmt.query([]).unwrap();

    let mut result: Vec<(i32, f64)> = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        result.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }

    assert_eq!(result, vec![(1, 450.0), (2, 200.0)]);
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));
}

#[cfg(target_os = "macos")]
#[test]
fn grouped_topk_hot_cache_is_visible_and_resettable() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE customer(c_custkey INTEGER, c_mktsegment VARCHAR);
             INSERT INTO customer VALUES (1, 'BUILDING'), (2, 'HOUSEHOLD'), (3, 'BUILDING');
             CREATE TABLE orders(o_orderkey INTEGER, o_custkey INTEGER, o_orderdate DATE, o_shippriority INTEGER);
             INSERT INTO orders VALUES
               (100, 1, DATE '1995-03-10', 0),
               (101, 2, DATE '1995-03-20', 1),
               (102, 3, DATE '1995-03-05', 0),
               (103, 1, DATE '1995-04-01', 2);
             CREATE TABLE lineitem(l_orderkey INTEGER, l_extendedprice DOUBLE, l_discount DOUBLE, l_shipdate DATE);
             INSERT INTO lineitem VALUES
               (100, 1000.0, 0.05, DATE '1995-03-30'),
               (100, 500.0,  0.10, DATE '1995-04-01'),
               (101, 200.0,  0.00, DATE '1995-03-28'),
               (102, 300.0,  0.20, DATE '1995-03-26'),
               (103, 800.0,  0.15, DATE '1995-03-30');",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('customer')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('orders')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('lineitem')")
        .unwrap();

    let sql = "CALL gpu_processing('SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue, \
               o_orderdate, o_shippriority \
               FROM customer \
               JOIN orders ON c_custkey = o_custkey \
               JOIN lineitem ON l_orderkey = o_orderkey \
               WHERE c_mktsegment = ''BUILDING'' \
               AND o_orderdate < DATE ''1995-03-25'' \
               AND l_shipdate > DATE ''1995-03-25'' \
               GROUP BY l_orderkey, o_orderdate, o_shippriority \
               ORDER BY revenue DESC \
               LIMIT 10')";

    let mut first = connection.prepare(sql).unwrap();
    let mut rows = first.query([]).unwrap();
    while rows.next().unwrap().is_some() {}
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));

    let (grouped_topk_entries_after_first, query_result_entries_after_first): (i64, i64) =
        connection
            .query_row("CALL gpu_cache_status()", [], |row| {
                Ok((row.get(15)?, row.get(17)?))
            })
            .unwrap();
    assert!(grouped_topk_entries_after_first >= 1);
    assert!(query_result_entries_after_first >= 1);

    let mut second = connection.prepare(sql).unwrap();
    let mut rows = second.query([]).unwrap();
    while rows.next().unwrap().is_some() {}
    assert_eq!(
        last_execution_mode().as_deref(),
        Some("metal_hot_query_cache")
    );

    let (status_mode, grouped_topk_entries_after_second, query_result_entries_after_second): (
        String,
        i64,
        i64,
    ) = connection
        .query_row("CALL gpu_cache_status()", [], |row| {
            Ok((row.get(0)?, row.get(15)?, row.get(17)?))
        })
        .unwrap();
    assert_eq!(status_mode, "metal_hot_query_cache");
    assert!(grouped_topk_entries_after_second >= 1);
    assert!(query_result_entries_after_second >= 1);

    let reset_status: String = connection
        .query_row("CALL gpu_cache_reset()", [], |row| row.get(0))
        .unwrap();
    assert_eq!(reset_status, "reset");

    let (reset_mode, reset_grouped_topk_entries, reset_query_result_entries): (String, i64, i64) =
        connection
            .query_row("CALL gpu_cache_status()", [], |row| {
                Ok((row.get(0)?, row.get(15)?, row.get(17)?))
            })
            .unwrap();
    assert_eq!(reset_mode, "");
    assert_eq!(reset_grouped_topk_entries, 0);
    assert_eq!(reset_query_result_entries, 0);
}

#[cfg(target_os = "macos")]
#[test]
fn single_table_gpu_query_hot_cache_is_visible() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE orders(key INTEGER, price DOUBLE, discount DOUBLE);
             INSERT INTO orders VALUES
               (1, 100.0, 0.05),
               (1, 200.0, 0.10),
               (2, 50.0, 0.0),
               (2, 150.0, 0.20);",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('orders')")
        .unwrap();

    let sql = "CALL gpu_processing('SELECT key, SUM(price * (1 - discount)) AS revenue \
               FROM orders GROUP BY key ORDER BY key')";

    let mut first = connection.prepare(sql).unwrap();
    let mut rows = first.query([]).unwrap();
    while rows.next().unwrap().is_some() {}
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));

    let query_result_entries_after_first: i64 = connection
        .query_row("CALL gpu_cache_status()", [], |row| row.get(17))
        .unwrap();
    assert!(query_result_entries_after_first >= 1);

    let mut second = connection.prepare(sql).unwrap();
    let mut rows = second.query([]).unwrap();
    while rows.next().unwrap().is_some() {}
    assert_eq!(
        last_execution_mode().as_deref(),
        Some("metal_hot_query_cache")
    );

    let (status_mode, query_result_entries_after_second): (String, i64) = connection
        .query_row("CALL gpu_cache_status()", [], |row| {
            Ok((row.get(0)?, row.get(17)?))
        })
        .unwrap();
    assert_eq!(status_mode, "metal_hot_query_cache");
    assert!(query_result_entries_after_second >= 1);
}

#[cfg(target_os = "macos")]
#[test]
fn generic_plan_executes_grouped_aggregate() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE sales(category VARCHAR, amount INTEGER);
             INSERT INTO sales VALUES ('A', 10), ('B', 20), ('A', 30), ('B', 40), ('A', 50);",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('sales')")
        .unwrap();

    let mut stmt = connection
        .prepare(
            "CALL gpu_processing('SELECT category, SUM(amount) AS total FROM sales GROUP BY category ORDER BY category')",
        )
        .unwrap();
    let mut rows = stmt.query([]).unwrap();

    let mut result: Vec<(String, f64)> = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        result.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }
    result.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].0, "A");
    assert!((result[0].1 - 90.0).abs() < 0.1); // 10+30+50
    assert_eq!(result[1].0, "B");
    assert!((result[1].1 - 60.0).abs() < 0.1); // 20+40
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));
}

#[test]
fn generic_plan_executes_grouped_mixed_aggregates() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE sales(category VARCHAR, amount INTEGER);
             INSERT INTO sales VALUES ('A', 10), ('B', 20), ('A', 30), ('B', 40), ('A', 50);",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('sales')")
        .unwrap();

    let mut stmt = connection
        .prepare(
            "CALL gpu_processing('SELECT SUM(amount) AS total, category, COUNT(*) AS cnt, \
             MIN(amount) AS lo, MAX(amount) AS hi, AVG(amount) AS avg \
             FROM sales GROUP BY category ORDER BY category')",
        )
        .unwrap();
    let mut rows = stmt.query([]).unwrap();

    let mut result: Vec<(f64, String, f64, f64, f64, f64)> = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        result.push((
            row.get(0).unwrap(),
            row.get(1).unwrap(),
            row.get(2).unwrap(),
            row.get(3).unwrap(),
            row.get(4).unwrap(),
            row.get(5).unwrap(),
        ));
    }

    assert_eq!(result.len(), 2);
    assert_eq!(result[0].1, "A");
    assert_eq!(result[0].0, 90.0);
    assert_eq!(result[0].2, 3.0);
    assert_eq!(result[0].3, 10.0);
    assert_eq!(result[0].4, 50.0);
    assert_eq!(result[0].5, 30.0);
    assert_eq!(result[1].1, "B");
    assert_eq!(result[1].0, 60.0);
    assert_eq!(result[1].2, 2.0);
    assert_eq!(result[1].3, 20.0);
    assert_eq!(result[1].4, 40.0);
    assert_eq!(result[1].5, 30.0);
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));
}

#[cfg(target_os = "macos")]
#[test]
fn gpu_grouped_aggregate_single_integer_key() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE data(key INTEGER, price DOUBLE, discount DOUBLE);
             INSERT INTO data VALUES
               (1, 100.0, 0.05), (1, 200.0, 0.10),
               (2, 50.0, 0.0), (2, 150.0, 0.20),
               (3, 300.0, 0.15);",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('data')")
        .unwrap();

    // Single integer key + SUM(expression) → should use GPU hash aggregate kernel.
    // key=1: 100*0.95 + 200*0.90 = 95 + 180 = 275
    // key=2: 50*1.0 + 150*0.80 = 50 + 120 = 170
    // key=3: 300*0.85 = 255
    let mut stmt = connection
        .prepare(
            "CALL gpu_processing('SELECT key, SUM(price * (1 - discount)) AS revenue FROM data GROUP BY key ORDER BY key')",
        )
        .unwrap();
    let mut rows = stmt.query([]).unwrap();

    let mut result: Vec<(i32, f64)> = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        result.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].0, 1);
    assert!((result[0].1 - 275.0).abs() < 1.0);
    assert_eq!(result[1].0, 2);
    assert!((result[1].1 - 170.0).abs() < 1.0);
    assert_eq!(result[2].0, 3);
    assert!((result[2].1 - 255.0).abs() < 1.0);
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));
}

#[cfg(target_os = "macos")]
#[test]
fn gpu_grouped_aggregate_single_integer_key_two_sums() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE data(key INTEGER, price DOUBLE, discount DOUBLE);
             INSERT INTO data VALUES
               (1, 100.0, 0.05), (1, 200.0, 0.10),
               (2, 50.0, 0.0), (2, 150.0, 0.20),
               (3, 300.0, 0.15);",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('data')")
        .unwrap();

    let mut stmt = connection
        .prepare(
            "CALL gpu_processing('SELECT key, SUM(price) AS gross, \
             SUM(price * (1 - discount)) AS net FROM data GROUP BY key ORDER BY key')",
        )
        .unwrap();
    let mut rows = stmt.query([]).unwrap();

    let mut result: Vec<(i32, f64, f64)> = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        result.push((
            row.get(0).unwrap(),
            row.get(1).unwrap(),
            row.get(2).unwrap(),
        ));
    }

    assert_eq!(
        result,
        vec![(1, 300.0, 275.0), (2, 200.0, 170.0), (3, 300.0, 255.0),]
    );
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));
}

#[cfg(target_os = "macos")]
#[test]
fn gpu_grouped_aggregate_composite_key_multi_state() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE data(region INTEGER, category INTEGER, amount DOUBLE);
             INSERT INTO data VALUES
               (1, 10, 100.0), (1, 10, 40.0),
               (1, 20, 30.0), (1, 20, 50.0),
               (2, 10, 80.0), (2, 10, 20.0);",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('data')")
        .unwrap();

    let mut stmt = connection
        .prepare(
            "CALL gpu_processing('SELECT region, category, COUNT(*) AS cnt, SUM(amount) AS total, \
             MIN(amount) AS lo, MAX(amount) AS hi, AVG(amount) AS avg \
             FROM data GROUP BY region, category ORDER BY region, category')",
        )
        .unwrap();
    let mut rows = stmt.query([]).unwrap();

    let mut result: Vec<(i32, i32, f64, f64, f64, f64, f64)> = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        result.push((
            row.get(0).unwrap(),
            row.get(1).unwrap(),
            row.get(2).unwrap(),
            row.get(3).unwrap(),
            row.get(4).unwrap(),
            row.get(5).unwrap(),
            row.get(6).unwrap(),
        ));
    }

    assert_eq!(
        result,
        vec![
            (1, 10, 2.0, 140.0, 40.0, 100.0, 70.0),
            (1, 20, 2.0, 80.0, 30.0, 50.0, 40.0),
            (2, 10, 2.0, 100.0, 20.0, 80.0, 50.0),
        ]
    );
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));
}

#[cfg(target_os = "macos")]
#[test]
fn generic_plan_executes_grouped_aggregate_with_expression() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE orders(key INTEGER, price DOUBLE, discount DOUBLE);
             INSERT INTO orders VALUES (1, 100.0, 0.05), (1, 200.0, 0.10), (2, 50.0, 0.0), (2, 150.0, 0.20);",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('orders')")
        .unwrap();

    // SUM(price * (1 - discount)) grouped by key
    // key=1: 100*0.95 + 200*0.90 = 95 + 180 = 275
    // key=2: 50*1.0 + 150*0.80 = 50 + 120 = 170
    let mut stmt = connection
        .prepare(
            "CALL gpu_processing('SELECT key, SUM(price * (1 - discount)) AS revenue FROM orders GROUP BY key ORDER BY key')",
        )
        .unwrap();
    let mut rows = stmt.query([]).unwrap();

    let mut result: Vec<(f64, f64)> = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        result.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }
    // key column comes through as f64 (since it's in a grouped aggregate context)
    // Actually key is Int32, should come through as Int32 for group-by columns
    assert_eq!(result.len(), 2);
    assert!((result[0].1 - 275.0).abs() < 1.0);
    assert!((result[1].1 - 170.0).abs() < 1.0);
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));
}

#[cfg(target_os = "macos")]
#[test]
fn generic_plan_executes_two_table_grouped_aggregate() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE departments(id INTEGER, region INTEGER);
             INSERT INTO departments VALUES (1, 10), (2, 10), (3, 20);
             CREATE TABLE employees(dept_id INTEGER, salary DOUBLE);
             INSERT INTO employees VALUES
                 (1, 100.0), (1, 150.0), (2, 200.0), (3, 90.0), (3, 110.0);",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('departments')")
        .unwrap();
    connection
        .execute_batch("CALL gpu_register_table('employees')")
        .unwrap();

    let mut stmt = connection
        .prepare(
            "CALL gpu_processing('SELECT d.region, SUM(e.salary) AS payroll
                                  FROM departments d
                                  JOIN employees e ON d.id = e.dept_id
                                  GROUP BY d.region
                                  ORDER BY d.region')",
        )
        .unwrap();
    let mut rows = stmt.query([]).unwrap();

    let mut result: Vec<(i32, f64)> = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        result.push((row.get(0).unwrap(), row.get(1).unwrap()));
    }

    assert_eq!(result, vec![(10, 450.0), (20, 200.0)]);
    assert_eq!(last_execution_mode().as_deref(), Some("metal_generic_plan"));
}

#[test]
fn gpu_processing_falls_back_for_unpinned_tables() {
    let _guard = test_lock();
    reset_runtime_state_for_tests();
    let connection = Connection::open_in_memory().unwrap();
    register_extension(&connection).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE numbers(value INTEGER);
             INSERT INTO numbers VALUES (1), (2), (3);",
        )
        .unwrap();
    connection
        .execute_batch("CALL gpu_buffer_init('64 KB', '128 KB', pinned_memory_size = '256 KB')")
        .unwrap();

    // No gpu_register_table — should fall back to DuckDB CPU.
    let mut stmt = connection
        .prepare("CALL gpu_processing('SELECT value FROM numbers WHERE value > 1')")
        .unwrap();
    let mut rows = stmt.query([]).unwrap();

    let mut values: Vec<i32> = Vec::new();
    while let Some(row) = rows.next().unwrap() {
        values.push(row.get(0).unwrap());
    }
    values.sort();
    assert_eq!(values, vec![2, 3]);
    assert_eq!(
        last_execution_mode().as_deref(),
        Some("duckdb_cpu_fallback")
    );
}

fn test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

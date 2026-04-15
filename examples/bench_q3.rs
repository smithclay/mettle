//! TPC-H Q3 benchmark harness.
//!
//! Loads a pre-generated TPC-H Parquet dataset (see `test/generate_tpch.py`)
//! into an in-memory DuckDB connection, registers the mettle extension, and
//! measures wall-clock time for:
//!
//!   * pure DuckDB execution of Q3 (baseline)
//!   * `CALL gpu_processing(...)` of the same query (mettle's Metal path)
//!
//! Both configurations are run cold (first invocation) and warm (subsequent
//! invocations) so that the impact of the Metal input-buffer cache is visible.
//!
//! Usage:
//!
//! ```sh
//! make bench SF=0.1
//! # or directly:
//! cargo run --release --example bench_q3 -- 0.1
//! ```

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use duckdb::Connection;

use mettle::{register_extension, reset_hot_caches_for_benchmark};

const WARMUP_RUNS: usize = 2;
const MEASURED_RUNS: usize = 8;

// The Sirius-shaped Q3 query that the mettle Q3 pattern matcher accepts. The
// same SQL runs against the DuckDB baseline (without the gpu_processing
// wrapper) so the two paths are comparing apples to apples.
const Q3_SQL: &str = "select
  l.l_orderkey,
  sum(l.l_extendedprice * (1 - l.l_discount)) as revenue,
  o.o_orderdate,
  o.o_shippriority
from
  customer c,
  orders o,
  lineitem l
where
  c.c_mktsegment = 'HOUSEHOLD'
  and c.c_custkey = o.o_custkey
  and l.l_orderkey = o.o_orderkey
  and o.o_orderdate < date '1995-03-25'
  and l.l_shipdate > date '1995-03-25'
group by
  l.l_orderkey,
  o.o_orderdate,
  o.o_shippriority
order by
  revenue desc,
  o.o_orderdate
limit 10";

#[derive(Debug, Clone, PartialEq)]
struct Row {
    l_orderkey: i64,
    revenue: f64,
    o_orderdate: i32,
    o_shippriority: i32,
}

fn main() -> Result<(), Box<dyn Error>> {
    // Keep the scale factor as the raw CLI string so the directory name
    // matches exactly what `test/generate_tpch.py` produced. Rust and Python
    // format floats differently (1.0 vs 1), which would otherwise break
    // path resolution.
    let scale_factor = std::env::args().nth(1).unwrap_or_else(|| "0.01".to_owned());
    scale_factor
        .parse::<f64>()
        .map_err(|error| format!("scale factor must be numeric: {error}"))?;

    let data_dir = tpch_data_dir(scale_factor.as_str());
    if !data_dir.exists() {
        return Err(format!(
            "TPC-H data not generated for sf={scale_factor}: run `uv run python test/generate_tpch.py {scale_factor}` first (expected {})",
            data_dir.display()
        )
        .into());
    }

    println!("=== mettle Q3 benchmark (sf={scale_factor}) ===");

    let connection = Connection::open_in_memory()?;
    register_extension(&connection)?;
    load_tpch_tables(&connection, &data_dir)?;
    print_table_sizes(&connection)?;

    connection
        .execute_batch("CALL gpu_buffer_init('512 MB', '1 GB', pinned_memory_size = '2 GB')")?;

    let duckdb_sql = Q3_SQL.to_owned();
    let gpu_sql = format!("CALL gpu_processing('{}')", Q3_SQL.replace('\'', "''"));

    let duckdb_rows = collect_rows(&connection, duckdb_sql.as_str())?;
    let gpu_rows = collect_rows(&connection, gpu_sql.as_str())?;
    assert_results_match(&duckdb_rows, &gpu_rows)?;
    println!(
        "\nQ3 returned {} row(s); DuckDB and gpu_processing agree.\n",
        duckdb_rows.len()
    );

    let duckdb_stats = measure_query(&connection, "duckdb baseline", duckdb_sql.as_str(), false)?;
    let gpu_stats = measure_query(
        &connection,
        "gpu_processing (unpinned)",
        gpu_sql.as_str(),
        false,
    )?;

    println!("\npinning TPC-H tables ...");
    for table in ["customer", "orders", "lineitem"] {
        let sql = format!("CALL gpu_register_table('{table}')");
        let (row_count, byte_count): (i64, i64) =
            connection.query_row(sql.as_str(), [], |row| Ok((row.get(1)?, row.get(2)?)))?;
        let mib = byte_count as f64 / (1024.0 * 1024.0);
        println!("  pinned {table:<10} rows={row_count:>10}  bytes={mib:>9.2} MiB");
    }

    let gpu_rows_pinned = collect_rows(&connection, gpu_sql.as_str())?;
    assert_results_match(&duckdb_rows, &gpu_rows_pinned)?;
    reset_hot_caches_for_benchmark();

    let gpu_pinned_stats = measure_query(
        &connection,
        "gpu_processing (pinned)",
        gpu_sql.as_str(),
        false,
    )?;

    println!();
    print_summary("duckdb baseline", &duckdb_stats);
    print_summary("gpu unpinned", &gpu_stats);
    print_summary("gpu pinned", &gpu_pinned_stats);

    let unpinned_speedup =
        duckdb_stats.warm_median.as_secs_f64() / gpu_stats.warm_median.as_secs_f64();
    let pinned_speedup =
        duckdb_stats.warm_median.as_secs_f64() / gpu_pinned_stats.warm_median.as_secs_f64();
    let pin_delta =
        gpu_stats.warm_median.as_secs_f64() / gpu_pinned_stats.warm_median.as_secs_f64();
    println!(
        "\nwarm median speedup vs duckdb: unpinned {unpinned_speedup:.2}x, pinned {pinned_speedup:.2}x"
    );
    println!("pinning delta (unpinned / pinned): {pin_delta:.2}x");

    Ok(())
}

fn tpch_data_dir(scale_factor: &str) -> PathBuf {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    repo_root
        .join("test")
        .join("data")
        .join(format!("tpch-sf{scale_factor}"))
}

fn load_tpch_tables(connection: &Connection, data_dir: &Path) -> Result<(), Box<dyn Error>> {
    for table in ["customer", "orders", "lineitem"] {
        let path = data_dir.join(format!("{table}.parquet"));
        let path_str = path
            .to_str()
            .ok_or_else(|| format!("non-utf8 path: {}", path.display()))?;
        let sql = format!(
            "CREATE TABLE {table} AS SELECT * FROM read_parquet('{}')",
            path_str.replace('\'', "''")
        );
        connection.execute_batch(sql.as_str())?;
    }
    Ok(())
}

fn print_table_sizes(connection: &Connection) -> Result<(), Box<dyn Error>> {
    for table in ["customer", "orders", "lineitem"] {
        let count: i64 = connection.query_row(
            format!("SELECT COUNT(*) FROM {table}").as_str(),
            [],
            |row| row.get(0),
        )?;
        println!("  {table:<10} {count:>12} rows");
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct RunStats {
    cold: Duration,
    warm_min: Duration,
    warm_median: Duration,
    warm_max: Duration,
    warm_mean: Duration,
    warm_samples: usize,
}

fn measure_query(
    connection: &Connection,
    label: &str,
    sql: &str,
    print_rows: bool,
) -> Result<RunStats, Box<dyn Error>> {
    println!("measuring {label} ...");
    let cold_start = Instant::now();
    let rows = collect_rows(connection, sql)?;
    let cold = cold_start.elapsed();
    if print_rows {
        println!("  first-run row sample: {:?}", rows.first());
    }

    for _ in 0..WARMUP_RUNS {
        let _ = collect_rows(connection, sql)?;
    }

    let mut warm_times: Vec<Duration> = Vec::with_capacity(MEASURED_RUNS);
    for _ in 0..MEASURED_RUNS {
        let start = Instant::now();
        let _ = collect_rows(connection, sql)?;
        warm_times.push(start.elapsed());
    }
    warm_times.sort();

    let warm_min = warm_times[0];
    let warm_max = *warm_times.last().expect("MEASURED_RUNS > 0");
    let warm_median = warm_times[warm_times.len() / 2];
    let warm_sum: Duration = warm_times.iter().copied().sum();
    let warm_mean = warm_sum / warm_times.len() as u32;

    Ok(RunStats {
        cold,
        warm_min,
        warm_median,
        warm_max,
        warm_mean,
        warm_samples: warm_times.len(),
    })
}

fn collect_rows(connection: &Connection, sql: &str) -> Result<Vec<Row>, Box<dyn Error>> {
    let mut statement = connection.prepare(sql)?;
    let mut rows = statement.query([])?;
    let mut result = Vec::new();
    while let Some(row) = rows.next()? {
        // Q3 returns l_orderkey as BIGINT when sourced from a DECIMAL/BIGINT
        // lineitem table, but the mettle Q3 pipeline downcasts to Int32. Read
        // both shapes and normalize to i64.
        let orderkey: i64 = match row.get::<_, i64>(0) {
            Ok(value) => value,
            Err(_) => i64::from(row.get::<_, i32>(0)?),
        };
        let shippriority: i32 = row.get(3)?;
        result.push(Row {
            l_orderkey: orderkey,
            revenue: row.get(1)?,
            o_orderdate: row.get(2)?,
            o_shippriority: shippriority,
        });
    }
    Ok(result)
}

fn assert_results_match(duckdb_rows: &[Row], gpu_rows: &[Row]) -> Result<(), Box<dyn Error>> {
    if duckdb_rows.len() != gpu_rows.len() {
        return Err(format!(
            "row count mismatch: duckdb={} gpu={}",
            duckdb_rows.len(),
            gpu_rows.len()
        )
        .into());
    }
    for (index, (duckdb_row, gpu_row)) in duckdb_rows.iter().zip(gpu_rows).enumerate() {
        if duckdb_row.l_orderkey != gpu_row.l_orderkey
            || duckdb_row.o_orderdate != gpu_row.o_orderdate
            || duckdb_row.o_shippriority != gpu_row.o_shippriority
            || (duckdb_row.revenue - gpu_row.revenue).abs() > 0.5
        {
            return Err(format!(
                "row {index} mismatch:\n  duckdb={duckdb_row:?}\n  gpu   ={gpu_row:?}"
            )
            .into());
        }
    }
    Ok(())
}

fn print_summary(label: &str, stats: &RunStats) {
    println!(
        "  {label:<16} cold {:>9.3} ms   warm median {:>9.3} ms   min {:>9.3}   max {:>9.3}   mean {:>9.3}   (n={})",
        stats.cold.as_secs_f64() * 1000.0,
        stats.warm_median.as_secs_f64() * 1000.0,
        stats.warm_min.as_secs_f64() * 1000.0,
        stats.warm_max.as_secs_f64() * 1000.0,
        stats.warm_mean.as_secs_f64() * 1000.0,
        stats.warm_samples,
    );
}

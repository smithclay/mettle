use crate::error::{MettleError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferConfig {
    pub cache_region_bytes: u64,
    pub processing_region_bytes: u64,
    pub pinned_region_bytes: u64,
}

impl BufferConfig {
    pub fn parse(
        cache_region_size: &str,
        processing_region_size: &str,
        pinned_memory_size: Option<&str>,
    ) -> Result<Self> {
        let cache_region_bytes = parse_human_bytes(cache_region_size)?;
        let processing_region_bytes = parse_human_bytes(processing_region_size)?;
        let pinned_region_bytes = match pinned_memory_size {
            Some(size) => parse_human_bytes(size)?,
            None => cache_region_bytes
                .checked_add(processing_region_bytes)
                .ok_or_else(|| {
                    MettleError::InvalidBufferConfig(
                        "pinned memory default overflowed the configured budgets".to_owned(),
                    )
                })?,
        };

        if cache_region_bytes == 0 {
            return Err(MettleError::InvalidBufferConfig(
                "gpu_buffer_init requires a non-zero cache region".to_owned(),
            ));
        }

        if processing_region_bytes == 0 {
            return Err(MettleError::InvalidBufferConfig(
                "gpu_buffer_init requires a non-zero processing region".to_owned(),
            ));
        }

        if pinned_region_bytes == 0 {
            return Err(MettleError::InvalidBufferConfig(
                "gpu_buffer_init requires a non-zero pinned region".to_owned(),
            ));
        }

        Ok(Self {
            cache_region_bytes,
            processing_region_bytes,
            pinned_region_bytes,
        })
    }
}

pub fn to_i64(value: u64, label: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| {
        MettleError::InvalidBufferConfig(format!("{label} exceeded DuckDB BIGINT capacity"))
    })
}

fn parse_human_bytes(raw: &str) -> Result<u64> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(MettleError::InvalidBufferConfig(
            "buffer sizes cannot be empty".to_owned(),
        ));
    }

    let mut numeric = String::new();
    let mut unit = String::new();
    for ch in trimmed.chars() {
        if ch.is_ascii_digit() || ch == '.' {
            if !unit.is_empty() {
                return Err(MettleError::InvalidBufferConfig(format!(
                    "invalid size literal '{raw}'"
                )));
            }
            numeric.push(ch);
        } else if ch.is_ascii_whitespace() {
            continue;
        } else {
            unit.push(ch);
        }
    }

    if numeric.is_empty() {
        return Err(MettleError::InvalidBufferConfig(format!(
            "missing numeric value in '{raw}'"
        )));
    }

    let quantity = numeric.parse::<f64>().map_err(|_| {
        MettleError::InvalidBufferConfig(format!("invalid numeric value in '{raw}'"))
    })?;

    if !quantity.is_finite() || quantity < 0.0 {
        return Err(MettleError::InvalidBufferConfig(format!(
            "invalid size literal '{raw}'"
        )));
    }

    let multiplier = match unit.to_ascii_lowercase().as_str() {
        "" | "b" => 1_f64,
        "k" | "kb" | "kib" => 1024_f64,
        "m" | "mb" | "mib" => 1024_f64.powi(2),
        "g" | "gb" | "gib" => 1024_f64.powi(3),
        "t" | "tb" | "tib" => 1024_f64.powi(4),
        other => {
            return Err(MettleError::InvalidBufferConfig(format!(
                "unsupported size unit '{other}' in '{raw}'"
            )))
        }
    };

    let bytes = quantity * multiplier;
    if bytes > u64::MAX as f64 {
        return Err(MettleError::InvalidBufferConfig(format!(
            "size literal '{raw}' overflowed u64"
        )));
    }

    Ok(bytes.round() as u64)
}

#[cfg(test)]
mod tests {
    use super::{parse_human_bytes, BufferConfig};

    #[test]
    fn parses_human_sizes() {
        assert_eq!(parse_human_bytes("64").unwrap(), 64);
        assert_eq!(parse_human_bytes("64 KB").unwrap(), 64 * 1024);
        assert_eq!(parse_human_bytes("1.5 MB").unwrap(), 1_572_864);
        assert_eq!(parse_human_bytes("2gb").unwrap(), 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn defaults_pinned_budget_to_sum() {
        let config = BufferConfig::parse("1 KB", "2 KB", None).unwrap();
        assert_eq!(config.cache_region_bytes, 1024);
        assert_eq!(config.processing_region_bytes, 2048);
        assert_eq!(config.pinned_region_bytes, 3072);
    }
}

use std::fmt;

pub type Result<T> = std::result::Result<T, MettleError>;

#[derive(Debug)]
pub enum MettleError {
    InvalidBufferConfig(String),
    MetalUnavailable(String),
    NotInitialized(String),
    Processing(String),
}

impl fmt::Display for MettleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBufferConfig(message) => write!(f, "invalid buffer config: {message}"),
            Self::MetalUnavailable(message) => write!(f, "metal unavailable: {message}"),
            Self::NotInitialized(message) => write!(f, "not initialized: {message}"),
            Self::Processing(message) => write!(f, "processing error: {message}"),
        }
    }
}

impl std::error::Error for MettleError {}

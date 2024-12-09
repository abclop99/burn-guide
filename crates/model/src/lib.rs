//! The lib

pub mod data;
pub mod inference;
pub mod model;

#[cfg(feature = "train")]
pub mod training;

// src/config/loader.rs
//! Enhanced configuration loader with validation and hot reload

use crate::config::{constants::paths, schema_validator::SchemaValidator, SystemConfig};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, RwLock};
use std::thread;
use std::time::{Duration, SystemTime};

/// Configuration loader with hot reload capabilities
pub struct ConfigLoader {
    config_paths: Vec<PathBuf>,
    schema_validator: SchemaValidator,
    current_config: Arc<RwLock<SystemConfig>>,
    change_notifier: Option<mpsc::Sender<SystemConfig>>,
    _file_watcher: Option<notify::RecommendedWatcher>,
}

/// Configuration loading errors
#[derive(Debug)]
pub enum ConfigError {
    FileNotFound(String),
    ParseError(String),
    ValidationError(Vec<crate::config::schema_validator::ValidationError>),
    IoError(String),
    WatcherError(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::FileNotFound(path) => write!(f, "Configuration file not found: {}", path),
            ConfigError::ParseError(msg) => write!(f, "Configuration parse error: {}", msg),
            ConfigError::ValidationError(errors) => {
                write!(f, "Configuration validation errors: ")?;
                for error in errors {
                    write!(f, "\n  {}", error)?;
                }
                Ok(())
            }
            ConfigError::IoError(msg) => write!(f, "IO error: {}", msg),
            ConfigError::WatcherError(msg) => write!(f, "File watcher error: {}", msg),
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<std::io::Error> for ConfigError {
    fn from(err: std::io::Error) -> Self {
        ConfigError::IoError(err.to_string())
    }
}

impl From<toml::de::Error> for ConfigError {
    fn from(err: toml::de::Error) -> Self {
        ConfigError::ParseError(err.to_string())
    }
}

impl ConfigLoader {
    /// Create new configuration loader
    pub fn new() -> Self {
        Self {
            config_paths: Self::discover_config_paths(),
            schema_validator: SchemaValidator::new(),
            current_config: Arc::new(RwLock::new(SystemConfig::default())),
            change_notifier: None,
            _file_watcher: None,
        }
    }

    /// Create loader with custom paths
    pub fn with_paths(paths: Vec<PathBuf>) -> Self {
        Self {
            config_paths: paths,
            schema_validator: SchemaValidator::new(),
            current_config: Arc::new(RwLock::new(SystemConfig::default())),
            change_notifier: None,
            _file_watcher: None,
        }
    }

    /// Load system configuration with validation
    pub fn load_system_config(&mut self) -> Result<SystemConfig, ConfigError> {
        let config = self.load_and_merge_configs()?;

        // Update current config
        {
            let mut current = self.current_config.write().unwrap();
            *current = config.clone();
        }

        Ok(config)
    }

    /// Get current configuration
    pub fn get_current_config(&self) -> SystemConfig {
        self.current_config.read().unwrap().clone()
    }

    /// Setup hot reload with change notifications
    pub fn enable_hot_reload(
        &mut self,
        callback: impl Fn(SystemConfig) + Send + 'static,
    ) -> Result<(), ConfigError> {
        let (tx, rx) = mpsc::channel();
        self.change_notifier = Some(tx.clone());

        // Spawn thread to handle config changes
        let config_ref = self.current_config.clone();
        thread::spawn(move || {
            while let Ok(new_config) = rx.recv() {
                {
                    let mut current = config_ref.write().unwrap();
                    *current = new_config.clone();
                }
                callback(new_config);
            }
        });

        // Setup file watcher
        self.setup_file_watcher(tx)?;
        Ok(())
    }

    /// Reload configuration manually
    pub fn reload(&mut self) -> Result<SystemConfig, ConfigError> {
        let config = self.load_and_merge_configs()?;

        {
            let mut current = self.current_config.write().unwrap();
            *current = config.clone();
        }

        if let Some(ref notifier) = self.change_notifier {
            let _ = notifier.send(config.clone());
        }

        Ok(config)
    }

    /// Validate configuration without loading
    pub fn validate_config_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let toml_value: toml::Value = toml::from_str(&content)?;

        // Validate against schema
        self.schema_validator
            .validate_config(&toml_value)
            .map_err(ConfigError::ValidationError)?;

        // Validate dependencies
        self.schema_validator
            .validate_dependencies(&toml_value)
            .map_err(ConfigError::ValidationError)?;

        Ok(())
    }

    /// Export current configuration to file
    pub fn export_config<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError> {
        let config = self.get_current_config();
        let toml_content =
            toml::to_string_pretty(&config).map_err(|e| ConfigError::ParseError(e.to_string()))?;

        std::fs::write(path, toml_content)?;
        Ok(())
    }

    /// Get configuration file modification times
    pub fn get_config_timestamps(&self) -> Vec<(PathBuf, Option<SystemTime>)> {
        self.config_paths
            .iter()
            .map(|path| {
                let timestamp = std::fs::metadata(path)
                    .and_then(|meta| meta.modified())
                    .ok();
                (path.clone(), timestamp)
            })
            .collect()
    }

    fn load_and_merge_configs(&self) -> Result<SystemConfig, ConfigError> {
        let mut merged_config = toml::Value::Table(toml::value::Table::new());

        // Start with default configuration
        let default_config = toml::Value::try_from(&SystemConfig::default())
            .map_err(|e| ConfigError::ParseError(e.to_string()))?;
        self.merge_toml_values(&mut merged_config, default_config);

        // Load and merge each configuration file
        for config_path in &self.config_paths {
            if config_path.exists() {
                match self.load_config_file(config_path) {
                    Ok(file_config) => {
                        self.merge_toml_values(&mut merged_config, file_config);
                    }
                    Err(ConfigError::FileNotFound(_)) => continue, // Skip missing optional files
                    Err(e) => return Err(e),
                }
            }
        }

        // Apply environment variable overrides
        self.apply_environment_overrides(&mut merged_config);

        // Validate merged configuration
        self.schema_validator
            .validate_config(&merged_config)
            .map_err(ConfigError::ValidationError)?;

        self.schema_validator
            .validate_dependencies(&merged_config)
            .map_err(ConfigError::ValidationError)?;

        // Convert to SystemConfig
        let config: SystemConfig = merged_config.try_into().map_err(|e| {
            ConfigError::ParseError(format!("Failed to deserialize config: {:?}", e))
        })?;

        Ok(config)
    }

    fn load_config_file<P: AsRef<Path>>(&self, path: P) -> Result<toml::Value, ConfigError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(ConfigError::FileNotFound(path.display().to_string()));
        }

        let content = std::fs::read_to_string(path)?;
        let config: toml::Value = toml::from_str(&content)?;

        Ok(config)
    }

    fn merge_toml_values(&self, base: &mut toml::Value, overlay: toml::Value) {
        match (base, overlay) {
            (toml::Value::Table(base_table), toml::Value::Table(overlay_table)) => {
                for (key, value) in overlay_table {
                    if let Some(base_value) = base_table.get_mut(&key) {
                        self.merge_toml_values(base_value, value);
                    } else {
                        base_table.insert(key, value);
                    }
                }
            }
            (base_value, overlay_value) => {
                *base_value = overlay_value;
            }
        }
    }

    fn apply_environment_overrides(&self, config: &mut toml::Value) {
        use std::env;

        for (key, value) in env::vars() {
            if key.starts_with("EMG_") {
                let config_key = key
                    .strip_prefix("EMG_")
                    .unwrap()
                    .to_lowercase()
                    .replace('_', ".");

                if let Ok(parsed_value) = self.parse_env_value(&value) {
                    self.set_nested_value(config, &config_key, parsed_value);
                }
            }
        }
    }

    fn parse_env_value(&self, value: &str) -> Result<toml::Value, toml::de::Error> {
        // Try parsing as different types
        if let Ok(int_val) = value.parse::<i64>() {
            Ok(toml::Value::Integer(int_val))
        } else if let Ok(float_val) = value.parse::<f64>() {
            Ok(toml::Value::Float(float_val))
        } else if let Ok(bool_val) = value.parse::<bool>() {
            Ok(toml::Value::Boolean(bool_val))
        } else {
            Ok(toml::Value::String(value.to_string()))
        }
    }

    fn set_nested_value(&self, config: &mut toml::Value, path: &str, value: toml::Value) {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = config;

        for (i, part) in parts.iter().enumerate() {
            if i == parts.len() - 1 {
                // Last part - set the value
                if let toml::Value::Table(table) = current {
                    table.insert(part.to_string(), value.clone());
                }
            } else {
                // Intermediate part - ensure it's a table
                if let toml::Value::Table(table) = current {
                    let entry = table
                        .entry(part.to_string())
                        .or_insert_with(|| toml::Value::Table(toml::value::Table::new()));
                    current = entry;
                }
            }
        }
    }

    fn setup_file_watcher(&mut self, tx: mpsc::Sender<SystemConfig>) -> Result<(), ConfigError> {
        use notify::{DebouncedEvent, RecursiveMode, Watcher};

        let (watch_tx, watch_rx) = mpsc::channel();
        let mut watcher = notify::watcher(watch_tx, Duration::from_millis(500))
            .map_err(|e| ConfigError::WatcherError(e.to_string()))?;

        // Watch all config directories
        let mut watched_paths = std::collections::HashSet::new();
        for path in &self.config_paths {
            if let Some(parent) = path.parent() {
                if !watched_paths.contains(parent) {
                    let _ = watcher.watch(parent, RecursiveMode::NonRecursive);
                    watched_paths.insert(parent.to_path_buf());
                }
            }
        }

        let config_paths = self.config_paths.clone();
        let schema_validator = SchemaValidator::new();

        thread::spawn(move || {
            while let Ok(event) = watch_rx.recv() {
                match event {
                    DebouncedEvent::Write(path) | DebouncedEvent::Create(path) => {
                        if config_paths.iter().any(|p| p == &path) {
                            // Config file changed, reload
                            let mut loader = ConfigLoader::with_paths(config_paths.clone());
                            loader.schema_validator = schema_validator.clone();

                            match loader.load_and_merge_configs() {
                                Ok(new_config) => {
                                    let _ = tx.send(new_config);
                                }
                                Err(e) => {
                                    eprintln!("Failed to reload config: {}", e);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        });

        self._file_watcher = Some(watcher);
        Ok(())
    }

    fn discover_config_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // System-wide configuration
        paths.push(PathBuf::from(paths::SYSTEM_CONFIG_PATH));

        // User configuration
        if let Some(home_dir) = dirs::home_dir() {
            paths.push(home_dir.join(paths::USER_CONFIG_DIR).join("config.toml"));
        }

        // Local configurations (in order of precedence)
        paths.push(PathBuf::from(paths::DEFAULT_CONFIG_FILE));
        paths.push(PathBuf::from(paths::LOCAL_CONFIG_FILE));
        paths.push(PathBuf::from("config/local.toml"));

        paths
    }
}

impl Default for ConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}

// Cross-platform directory discovery
mod dirs {
    use std::path::PathBuf;

    pub fn home_dir() -> Option<PathBuf> {
        #[cfg(target_os = "windows")]
        {
            std::env::var_os("USERPROFILE").map(PathBuf::from)
        }
        #[cfg(not(target_os = "windows"))]
        {
            std::env::var_os("HOME").map(PathBuf::from)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_loader_creation() {
        let loader = ConfigLoader::new();
        assert!(!loader.config_paths.is_empty());
    }

    #[test]
    fn test_load_default_config() {
        let mut loader = ConfigLoader::new();
        let config = loader.load_system_config();
        assert!(config.is_ok());
    }

    #[test]
    fn test_config_file_validation() {
        let loader = ConfigLoader::new();

        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"
[system]
sampling_rate_hz = 2000
channel_count = 8

[hal]
device_type = "simulator"
        "#
        )
        .unwrap();

        assert!(loader.validate_config_file(temp_file.path()).is_ok());
    }

    #[test]
    fn test_invalid_config_validation() {
        let loader = ConfigLoader::new();

        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"
[system]
sampling_rate_hz = 50  # Too low
        "#
        )
        .unwrap();

        assert!(loader.validate_config_file(temp_file.path()).is_err());
    }

    #[test]
    fn test_environment_override() {
        unsafe {
            std::env::set_var("EMG_SYSTEM_SAMPLING_RATE_HZ", "4000");
        }

        let mut loader = ConfigLoader::new();
        let config = loader.load_system_config().unwrap();

        assert_eq!(config.system.sampling_rate_hz, 4000);

        unsafe {
            std::env::remove_var("EMG_SYSTEM_SAMPLING_RATE_HZ");
        }
    }

    #[test]
    fn test_config_export() {
        let loader = ConfigLoader::new();
        let temp_file = NamedTempFile::new().unwrap();

        assert!(loader.export_config(temp_file.path()).is_ok());

        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("[system]"));
    }
}

#!/bin/bash
# emg-core/scripts/generate_docs.sh

set -e

echo "🦀 Generating Rust API documentation..."

# Clean previous builds
cargo clean --doc

# Generate documentation with all features
cargo doc \
    --no-deps \
    --all-features \
    --document-private-items \
    --open

# Copy to central docs location
mkdir -p ../emg-docs/docs/api/rust/
cp -r target/doc/* ../emg-docs/docs/api/rust/

echo "✅ Rust documentation generated successfully"
echo "📍 Available at: ../emg-docs/docs/api/rust/emg_core/index.html"
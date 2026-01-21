#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/LennartPurucker/finetune_tabpfn_v2.git"
REPO_DIR="../external/finetune_tabpfn_v2"
COMMIT="144cf00"

mkdir -p external

if [ ! -d "$REPO_DIR" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

if [ -d ".git" ]; then
  git fetch origin
  git checkout "$COMMIT"
  rm -rf .git
fi

PYPROJECT="pyproject.toml"

if [ ! -f "$PYPROJECT" ]; then
  cat > "$PYPROJECT" <<'EOF'
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "finetuning-scripts"
version = "0.1.0"

[tool.setuptools]
packages = { find = {} }
EOF
else
  if ! grep -q "^\[build-system\]" "$PYPROJECT"; then
    cat >> "$PYPROJECT" <<'EOF'

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
EOF
  fi

  if ! grep -q "^\[project\]" "$PYPROJECT"; then
    cat >> "$PYPROJECT" <<'EOF'

[project]
name = "finetuning-scripts"
version = "0.1.0"
EOF
  fi

  if grep -q "^\[tool.setuptools\]" "$PYPROJECT"; then
    sed -i 's/^packages *=.*/packages = { find = {} }/' "$PYPROJECT"
  else
    cat >> "$PYPROJECT" <<'EOF'

[tool.setuptools]
packages = { find = {} }
EOF
  fi
fi

if [ ! -f "finetuning_scripts/__init__.py" ]; then
  touch finetuning_scripts/__init__.py
fi

echo "✔ Repo bereit unter $REPO_DIR (ohne .git)"
echo "✔ Commit $COMMIT ausgecheckt"
echo "✔ pyproject.toml korrekt für Subpackages konfiguriert"
echo "✔ finetuning_scripts inkl. Submodule installierbar"

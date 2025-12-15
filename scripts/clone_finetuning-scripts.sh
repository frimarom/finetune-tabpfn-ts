#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/LennartPurucker/finetune_tabpfn_v2.git"
REPO_DIR="../external/finetune_tabpfn_v2"
COMMIT="144cf00"

mkdir -p external

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

git fetch origin
git checkout "$COMMIT"

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
packages = ["finetuning_scripts"]
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

  if ! grep -q "^\[tool.setuptools\]" "$PYPROJECT"; then
    cat >> "$PYPROJECT" <<'EOF'

[tool.setuptools]
packages = ["finetuning_scripts"]
EOF
  fi
fi


if [ ! -f "finetuning_scripts/__init__.py" ]; then
  touch finetuning_scripts/__init__.py
fi

echo "✔ Repo bereit unter $REPO_DIR"
echo "✔ pyproject.toml konfiguriert"
echo "✔ finetuning_scripts als Package verfügbar"

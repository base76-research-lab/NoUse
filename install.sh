#!/usr/bin/env bash
# nouse install — kopplar systemd user-services och startar hjärnan
# Kör: bash install.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"

echo "🧠 nouse install"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Sätt Python-version och synka dependencies
_PY_VER="$(cat "$SCRIPT_DIR/.python-version" 2>/dev/null | tr -d '[:space:]' || echo '3.13')"
echo "→ Python $_PY_VER (via pyenv + uv)..."
cd "$SCRIPT_DIR"
uv sync --python "$_PY_VER"

# 2. Bygg Rust TDA-motorn
echo "→ Bygger Rust TDA-motor (persistent homology, H0/H1 Betti)..."
if [[ -d "$SCRIPT_DIR/crates/tda_engine" ]]; then
  cd "$SCRIPT_DIR/crates/tda_engine"
  _PY_BIN="$SCRIPT_DIR/.venv/bin/python"
  "$SCRIPT_DIR/.venv/bin/maturin" build --release --interpreter "$_PY_BIN" 2>&1 \
    | grep -E "(Finished|Built wheel|error)" || true
  # Installera hjulet i venv via uv
  _WHEEL=$(ls "$SCRIPT_DIR/crates/tda_engine/target/wheels/"tda_engine-*-cp313-*.whl 2>/dev/null | tail -1)
  if [[ -n "$_WHEEL" ]]; then
    _UV_BIN="$(command -v uv 2>/dev/null || ls "$HOME"/snap/code/*/local/bin/uv 2>/dev/null | tail -1 || echo "")"
    if [[ -n "$_UV_BIN" ]]; then
      "$_UV_BIN" pip install "$_WHEEL" --python "$_PY_BIN" --force-reinstall --quiet 2>/dev/null || true
    else
      "$_PY_BIN" -m pip install "$_WHEEL" --force-reinstall --quiet 2>/dev/null || true
    fi
    echo "  ✓ tda_engine installerad (Rust motor ~350x snabbare)"
  fi
  cd "$SCRIPT_DIR"
else
  echo "  ✗ crates/tda_engine saknas — Python-fallback används"
fi

# 4. Skapa systemd user-katalog
mkdir -p "$SYSTEMD_USER_DIR"

# 3. Kopiera service-filer
for f in \
  nouse-daemon.service \
  nouse-daemon.timer \
  nouse-backup.service \
  nouse-backup.timer \
  nouse-eval.service \
  nouse-eval.timer \
  nouse-watchdog.service \
  nouse-watchdog.timer; do
  if [[ ! -f "$SCRIPT_DIR/systemd/$f" ]]; then
    continue
  fi
  cp "$SCRIPT_DIR/systemd/$f" "$SYSTEMD_USER_DIR/$f"
  echo "  ✓ $f"
done

# 4. Ladda om systemd
systemctl --user daemon-reload

# 5. Aktivera och starta
systemctl --user enable --now nouse-daemon.timer
systemctl --user enable --now nouse-backup.timer
if [[ -f "$SYSTEMD_USER_DIR/nouse-watchdog.timer" ]]; then
  systemctl --user enable --now nouse-watchdog.timer
fi
if [[ -f "$SYSTEMD_USER_DIR/nouse-eval.timer" ]]; then
  systemctl --user enable --now nouse-eval.timer
fi

echo ""
echo "✅ nouse är installerat och körs i bakgrunden."
echo ""
echo "Kommandon:"
echo "  nouse daemon status        — se grafens nuläge"
echo "  nouse chat                 — prata med hjärnan"
echo "  nouse visualize            — öppna grafvisualiseringen"
echo "  journalctl --user -u nouse-daemon -f   — se brain-loop loggen live"
echo ""
echo "Systemd:"
echo "  systemctl --user status nouse-daemon"
echo "  systemctl --user status nouse-daemon.timer"
echo "  systemctl --user status nouse-eval.timer"
echo "  systemctl --user status nouse-watchdog.timer"

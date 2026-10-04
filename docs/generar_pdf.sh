#!/usr/bin/env bash
# Genera docs/procedimiento.pdf a partir de docs/procedimiento.md.
# Requisitos: pandoc, tectonic (o xelatex) y Node.js con npm.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x node_modules/.bin/mmdc ]]; then
  echo "Instalando mermaid-cli..."
  PUPPETEER_SKIP_DOWNLOAD=1 npm install --no-fund --no-audit --silent
fi

# Usa el Chrome instalado en lugar de descargar uno con puppeteer.
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [[ -z "${PUPPETEER_EXECUTABLE_PATH:-}" && -x "$CHROME" ]]; then
  export PUPPETEER_EXECUTABLE_PATH="$CHROME"
fi

MOTOR="${PDF_ENGINE:-$(command -v tectonic >/dev/null && echo tectonic || echo xelatex)}"

pandoc procedimiento.md \
  --from markdown \
  --lua-filter mermaid.lua \
  --pdf-engine "$MOTOR" \
  --syntax-highlighting tango \
  --output procedimiento.pdf

echo "PDF generado: docs/procedimiento.pdf"

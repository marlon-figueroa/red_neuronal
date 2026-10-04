-- Filtro de pandoc: convierte los bloques ```mermaid en figuras PDF con mermaid-cli.
-- Atributos opcionales del bloque: caption="..." (admite Markdown/LaTeX) y width="80%".

local mmdc = os.getenv("MMDC") or "node_modules/.bin/mmdc"
local carpeta = "build/mermaid"
local configuracion = "mermaid-config.json"

local function existe(ruta)
  local archivo = io.open(ruta, "r")
  if archivo then
    archivo:close()
    return true
  end
  return false
end

local function leer(ruta)
  local archivo = assert(io.open(ruta, "r"))
  local contenido = archivo:read("a")
  archivo:close()
  return contenido
end

local function renderizar(codigo)
  os.execute("mkdir -p " .. carpeta)
  local base = carpeta .. "/" .. pandoc.utils.sha1(codigo .. leer(configuracion))
  local salida = base .. ".pdf"
  if existe(salida) then
    return salida
  end

  local fuente = assert(io.open(base .. ".mmd", "w"))
  fuente:write(codigo)
  fuente:close()

  local comando = string.format(
    '"%s" -i "%s" -o "%s" -c "%s" --pdfFit -q',
    mmdc, base .. ".mmd", salida, configuracion
  )
  if not os.execute(comando) then
    error("mermaid-cli no pudo generar " .. base .. ".mmd")
  end
  return salida
end

function CodeBlock(bloque)
  if not bloque.classes:includes("mermaid") then
    return nil
  end

  local atributos = {}
  if bloque.attributes.width then
    atributos.width = bloque.attributes.width
  end
  local imagen = pandoc.Image({}, renderizar(bloque.text), "", pandoc.Attr("", {}, atributos))

  local titulo = {}
  if bloque.attributes.caption then
    titulo = pandoc.utils.blocks_to_inlines(pandoc.read(bloque.attributes.caption, "markdown").blocks)
  end
  return pandoc.Figure(pandoc.Plain({ imagen }), { pandoc.Plain(titulo) }, pandoc.Attr(bloque.identifier))
end

-- Generate the autoapi link for firecrown.<module_name>.<symbol>. The HTML site
-- links relative to docs/_static (the project root, at depth quarto.project.offset
-- from the page); a downloaded notebook links to Read the Docs.
local rtd_base = "https://firecrown.readthedocs.io/en/latest/"
local autoapi_base = quarto.project.offset .. "/../autoapi/"
if quarto.doc.is_format("ipynb") then
  autoapi_base = rtd_base .. "autoapi/"
end

function to_autoapi_link(module_name, symbol)
  local full_module_name = "firecrown." .. module_name
  local path_part = full_module_name:gsub("%.", "/")
  local path = autoapi_base .. path_part .. "/index.html#" .. full_module_name .. "." .. symbol
  return path
end

-- Function to generate a Pandoc Link
function LinkGen(target, symbol)
  local url = to_autoapi_link(target, symbol)
  return pandoc.Link(pandoc.Code(symbol), url)
end

-- Directory of the page relative to the project root: "" or "cookbook/".
local function page_dir()
  local root = quarto.project.directory
  local dir = pandoc.path.directory(quarto.doc.input_file)
  local rel = pandoc.path.make_relative(dir, root)
  if rel == "." or rel == "" then
    return ""
  end
  return rel .. "/"
end

-- Remove "dir/.." pairs from a relative path.
local function normalize(path)
  local parts = {}
  for part in path:gmatch("[^/]+") do
    if part == ".." and #parts > 0 then
      table.remove(parts)
    elseif part ~= "." then
      table.insert(parts, part)
    end
  end
  return table.concat(parts, "/")
end

-- In a downloaded notebook, a link to another page of the site (foo.qmd) points
-- to its rendered page on Read the Docs.
return {
  {
    Link = function(el)
      if quarto.doc.is_format("ipynb") and not el.target:match("^%a+://") then
        local page, anchor = el.target:match("^([^#]+)%.qmd(.*)$")
        if page then
          el.target = rtd_base .. "_static/" .. normalize(page_dir() .. page) .. ".html" .. anchor
          return el
        end
      end
      return nil
    end,
    Inline = function(el)
      if el.t == "Str" then
        local prefix, target, symbol, suffix = el.text:match("(.-)%[%[(.-)|(.-)%]%](.*)")
        if target and symbol then
          -- io.stderr:write("PREFIX: ", prefix, "\n") -- debug
          -- io.stderr:write("TARGET: ", target, "\n") -- debug
          -- io.stderr:write("SYMBOL: ", symbol, "\n") -- debug
          -- io.stderr:write("SUFFIX: ", suffix, "\n") -- debug
          local link = LinkGen(target, symbol) -- Assuming LinkGen returns a pandoc.Link
          -- Return a table of inlines: prefix, link, suffix
          local result = {}
          if prefix ~= "" then
            table.insert(result, pandoc.Str(prefix))
          end
          table.insert(result, link)
          if suffix ~= "" then
            table.insert(result, pandoc.Str(suffix))
          end
          return result
        end
      end
      return el
    end
  }
}

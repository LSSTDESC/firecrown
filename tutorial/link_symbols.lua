-- link_symbols.lua

-- This script is a Quarto Lua filter.
-- It reads a JSON file containing a map of symbols to URLs.
-- It then walks the Pandoc AST and replaces occurrences of these symbols
-- in text with hyperlinks.

local json = require("json")
local stringify = require("pandoc.utils").stringify

-- 1. Read the symbol map
local symbols_file = "docs/_static/symbol_map.json"
local file = io.open(symbols_file, "r")
if not file then
  -- If the file doesn't exist, we can't do anything.
  -- This might happen during a clean build. We just print a warning
  -- and return, leaving the AST unchanged.
  -- print("Warning: Could not open symbol map file: " .. symbols_file)
  return {}
end

local content = file:read("*a")
file:close()
local symbol_map = json.decode(content)
if not symbol_map then
  -- print("Warning: Could not parse symbol map file: " .. symbols_file)
  return {}
end

-- 2. Create a function to be applied to each element of the AST
function Str(el)
  -- We are looking for text nodes that might contain our symbols.
  -- The symbol could be part of a larger string, so we need to search.
  -- This is a simple approach. A more robust one would use regex.
  -- Pandoc's Lua environment doesn't have a built-in regex library,
  -- so we do a simple string find.
  
  -- To avoid replacing parts of words, we look for symbols enclosed in backticks
  -- like `firecrown.parameters.Parameter`. The `Str` element's text `el.text`
  -- will just be `firecrown.parameters.Parameter` if the markdown is ```...```.
  
  -- This filter will run on `Code` elements, not `Str` elements if the
  -- markdown is ` `code` `.

  return el -- Return the element unchanged, we will use the Code filter instead
end

function Code(el)
  -- el.text is the content of the code span
  local symbol = el.text
  local url = symbol_map[symbol]
  
  if url then
    -- If we have a match in our symbol map, replace this `Code` element
    -- with a `Link` element.
    -- The link text will be a `Code` element itself to preserve formatting.
    return pandoc.Link(
      {pandoc.Code(el.attr, symbol)}, -- Link text
      url,                             -- Link URL
      ""                               -- Link title
    )
  end

  -- If no match, return the element unchanged.
  return el
end


-- 3. Return the filter definition
-- We apply the `Code` function to all `Code` elements in the document.
return {
  { Code = Code }
}

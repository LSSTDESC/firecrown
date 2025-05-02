-- Function to generate autoapi link
function to_autoapi_link(module_name, symbol)
  local full_module_name = "firecrown." .. module_name
  local path_part = full_module_name:gsub("%.", "/")
  local path = "../autoapi/" .. path_part .. "/index.html#" .. full_module_name .. "." .. symbol
  return path
end

-- Function to generate a Pandoc Link
function LinkGen(target, symbol)
  local url = to_autoapi_link(target, symbol)
  return pandoc.Link(pandoc.Code(symbol), url)
end

return {
  {
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

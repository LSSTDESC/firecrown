-- Function to generate autoapi link
function to_autoapi_link(classname)
  local full_classname = "firecrown." .. classname
  local prefix, suffix = full_classname:match("^(.*)%.([^%.]+)$")
  if not prefix then
    io.stderr:write("ERROR: symbol name not found: ", full_classname, "\n")
    return "", ""
  end
  local path_part = prefix:gsub("%.", "/")
  local path = "../autoapi/" .. path_part .. "/index.html#" .. full_classname
  return path, suffix
end

-- Function to generate a Pandoc Link
function LinkGen(target)
  local url, symbol = to_autoapi_link(target)
  return pandoc.Link(pandoc.Code(symbol), url)
end

return {
  {
    Inline = function(el)
      if el.t == "Str" then
        local content = el.text:match("^%[%[(.-)%]%]")
        if content then
          -- io.stderr:write("STR: ", el.text, "\n")   -- debug
          -- io.stderr:write("MATCH: ", content, "\n") -- debug
          return LinkGen(content)
        end
      end
      return el
    end
  }
}

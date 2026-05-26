-- version_filter.lua
-- Quarto Lua filter to replace version placeholders with actual version
-- Gets version from metadata variable 'firecrown-version'

function Meta(m)
  -- Get version from metadata (passed by Makefile) or environment variable
  local version = m["firecrown-version"]
  
  if not version then
    version = os.getenv("FIRECROWN_VERSION")
  end
  
  if not version or version == "" then
    version = "development"
  end
  
  -- Convert to string if it's a Pandoc element
  if type(version) == "table" then
    version = pandoc.utils.stringify(version)
  else
    version = tostring(version)
  end
  
  -- Process subtitle if it exists
  if m.subtitle then
    -- Process the subtitle content
    local subtitle = m.subtitle
    if type(subtitle) == "table" then
      -- If it's a list of inline elements, process each one
      local result = pandoc.List()
      for i, el in ipairs(subtitle) do
        if el.t == "Str" then
          -- Replace both the env shortcode and our custom placeholder
          el.text = el.text:gsub("{{%s*<%s*env%s+FIRECROWN_VERSION%s*>%s*}}", version)
          el.text = el.text:gsub("{firecrown%-version}", version)
        end
        result:insert(el)
      end
      m.subtitle = result
    end
  end
  
  return m
end

return {
  { Meta = Meta }
}

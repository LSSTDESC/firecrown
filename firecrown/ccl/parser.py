# # extract sources
# sources = {}
# for name, keys in data.get('sources', {}).items():
#     if keys['kind'] in ['ClTracerLensing', 'ClTracerNumberCounts']:
#         sources[name] = parse_ccl_source(**keys)
#     else:
#         raise ValueError(
#             "Source type '%s' not recognized for source '%s'!" % (
#                 name, keys['type']))
# data['sources'] = sources

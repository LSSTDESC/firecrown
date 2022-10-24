import sacc
saccfile = "./examples/des_y1_3x2pt/des_y1_3x2pt_sacc_data_TATT.fits"
sacc_data = sacc.Sacc.load_fits(saccfile)
print(sacc_data.get_data_types())

for d in sacc_data.get_data_points():
	d.to_table(sacc_data.get_data_points())
	print(d)

'''
for t in sacc_data.get_data_types():
	#sacc.to_table(sacc_data)
	tracers = sacc_data.get_tracer_combinations(t)
	for tracer in tracers:
		data = sacc_data.get_mean(t, tracer)
		sacc.data_types.parse_data_type_name(sacc_data)#tracer.to_table()
		#print(data)
'''
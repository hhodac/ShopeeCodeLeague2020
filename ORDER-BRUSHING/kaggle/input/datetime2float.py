import datetime as dt
import re
from datetime import datetime
from time import mktime


inp = "order_brush_order.csv"
outp = "order_brush_float.csv"
f_csv = open(inp,"r")
f_out = open(outp,"wt")

line = f_csv.readline()
epoch = dt.datetime(2000, 1, 1,0,0,0)

#for t in [(d - epoch).total_seconds() for d in times]:
#    print('%.6f' % t)
while (line):
	line = line.rstrip()
	cells = line.split(',')
	datetime_str = cells[-1]
	#print (datetime_str)
	if (re.search('event_time', line)):
		t = 0
		f_out.write(line +',datetime_float\n')
	else:
		datetime_object = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
		t = mktime(datetime_object.timetuple())
		f_out.write(line + ',' + str(t) +'\n')
	line = f_csv.readline()
	
f_csv.close()
f_out.close()